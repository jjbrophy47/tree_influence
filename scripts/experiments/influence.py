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
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import intent
import util
from config import exp_args


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


def remove_and_evaluate(objective, ranking, tree,
                        X_train, y_train, X_test, y_test,
                        remove_frac_list, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_loss

    # result container
    result = {}
    result['loss'] = np.full(len(remove_frac_list), np.nan, dtype=np.float32)

    res = eval_fn(objective, tree, X_test, y_test, logger, prefix=f'{0:>5}: {0:>5.3f}%')
    result['loss'][0] = res['loss']

    for i, remove_frac in enumerate(remove_frac_list):
        n_remove = int(X_train.shape[0] * remove_frac)

        new_X_train = np.delete(X_train, ranking[:n_remove], axis=0)
        new_y_train = np.delete(y_train, ranking[:n_remove])

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            new_tree = clone(tree).fit(new_X_train, new_y_train)

            prefix = f'{i + 1:>5}: {remove_frac * 100:>5.3f}%'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)
            result['loss'][i] = res['loss']

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
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # compute influence
    start2 = time.time()
    influence = explainer.get_local_influence(X_test[test_idxs], y_test[test_idxs])
    inf_time = time.time() - start2

    logger.info(f'[INFO] explainer influence time: {inf_time:.5f}s')
    logger.info(f'[INFO] total time: {time.time() - begin:.5f}s')

    # get ranking
    ranking = np.argsort(influence, axis=0)[::-1]  # shape=(no. train, no. test)

    # get no. jobs to run in parallel
    if args.n_jobs == -1:
        n_jobs = joblib.cpu_count()

    else:
        assert args.n_jobs >= 1
        n_jobs = min(args.n_jobs, joblib.cpu_count())

    logger.info(f'\n[INFO] no. jobs: {n_jobs:,}')

    with joblib.Parallel(n_jobs=n_jobs) as parallel:

        n_finish = 0
        n_remain = len(test_idxs)

        res_list = []

        while n_remain > 0:
            n = min(min(10, n_jobs), n_remain)

            results = parallel(joblib.delayed(remove_and_evaluate)
                                             (objective, ranking[:, n_finish + i], tree,
                                              X_train, y_train, X_test[[idx]], y_test[[idx]],
                                              args.remove_frac, logger)
                                              for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

            # synchronization barrier
            res_list += results

            n_finish += n
            n_remain -= n

            cum_time = time.time() - start
            logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                        f', cum. time: {cum_time:.3f}s')

        # combine results from each test example
        result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)

    # store ALL train and test predictions
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
    result['remove_frac'] = np.array(args.remove_frac, dtype=np.float32)
    result['influence'] = influence
    result['ranking'] = ranking
    result['test_idxs'] = test_idxs
    result['y_test_pred'] = y_test_pred
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    exp_hash = util.dict_to_hash(exp_dict)

    # special cases
    if args.method in ['leaf_inf', 'leaf_refit']:
        if args.tree_type == 'lgb' and args.dataset == 'flight_delays':
            args.leaf_inf_atol = 1e-1
        elif args.tree_type == 'cb' and args.dataset in ['bean', 'naval']:
            args.leaf_inf_atol = 1e-1

    # get unique hash for the explainer
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


if __name__ == '__main__':
    main(exp_args.get_influence_args().parse_args())
