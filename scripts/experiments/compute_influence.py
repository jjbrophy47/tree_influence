"""
Compute global or local influence.
"""
import os
import sys
import time
import argparse
import resource
from datetime import datetime

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import intent
import util


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

    # compute infuence
    start = time.time()

    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start

    if args.inf_obj == 'global':
        start2 = time.time()
        influence = explainer.get_global_influence(X=X_test, y=y_test)
        inf_time = time.time() - start2

    else:

        # randomly select test instances to compute influence values for
        if args.inf_obj == 'local':
            avail_idxs = np.arange(X_test.shape[0])
            test_idxs = select_elements(avail_idxs, rng, n=args.n_test)

        elif args.inf_obj == 'local_correct' and objective != 'regression':
            y_pred = tree.predict(X_test)
            correct_idxs = np.where(y_pred == y_test)[0]
            n_test = min(args.n_test, len(correct_idxs))
            test_idxs = select_elements(correct_idxs, rng, n=n_test)

        else:
            assert args.inf_obj == 'local_incorrect' and objective != 'regression'
            y_pred = tree.predict(X_test)
            incorrect_idxs = np.where(y_pred != y_test)[0]
            n_test = min(args.n_test, len(incorrect_idxs))
            test_idxs = select_elements(incorrect_idxs, rng, n=n_test)

        start2 = time.time()
        influence = explainer.get_local_influence(X_test[test_idxs], y_test[test_idxs])
        inf_time = time.time() - start2

    # store ALL train and test predictions
    if objective == 'regression':
        y_train_pred = tree.predict(X_train).reshape(-1, 1)
        y_test_pred = tree.predict(X_test).reshape(-1, 1)

    elif objective == 'binary':
        y_test_pred = tree.predict_proba(X_test)[:, 1].reshape(-1, 1)
        y_train_pred = tree.predict_proba(X_train)[:, 1].reshape(-1, 1)

    else:
        assert objective == 'multiclass'
        y_test_pred = tree.predict_proba(X_test)
        y_train_pred = tree.predict_proba(X_train)

    # display influence
    logger.info(f'\ninfluence: {influence}, shape: {influence.shape}')

    logger.info(f'fit time: {fit_time:.5f}s')
    logger.info(f'compute time: {inf_time:.5f}s')

    # save results
    result = {}
    result['influence'] = influence
    result['test_idxs'] = test_idxs if args.inf_obj != 'global' else ''
    result['y_train_pred'] = y_train_pred
    result['y_test_pred'] = y_test_pred
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get method params and unique settings hash
    params, hash_str = util.explainer_params_to_dict(args.method, vars(args))

    # special cases
    if args.method == 'leaf_influence':
        if args.dataset == 'flight_delays':
            params['atol'] = 1e-1
        elif args.dataset == 'twitter':
            params['atol'] = 1e-0

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'rs_{args.random_state}',
                           args.inf_obj,
                           f'{args.method}_{hash_str}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, params, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/influence/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='synthetic_regression')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Explainer settings
    parser.add_argument('--method', type=str, default='random')

    parser.add_argument('--use_leaf', type=int, default=1)  # BoostIn
    parser.add_argument('--local_op', type=str, default='normal')  # BoostIn

    parser.add_argument('--update_set', type=int, default=-1)  # LeafInfluence

    parser.add_argument('--similarity', type=str, default='dot_prod')  # Similarity

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex & similarity
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    # Experiment settings
    parser.add_argument('--inf_obj', type=str, default='global')
    parser.add_argument('--n_test', type=int, default=100)  # local

    args = parser.parse_args()
    main(args)
