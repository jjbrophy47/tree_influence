"""
Compute local influence w/ influence reestimation.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime
from datetime import timedelta

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # tree_influence
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import tree_influence
import util
from influence import get_special_case_tol
from influence import select_elements
from config import exp_args


def remove_and_reinfluence(objective, tree, method, params,
                           X_train, y_train, X_test, y_test,
                           remove_frac, strategy, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_loss

    # get list of remove fractions
    total_n_remove = round(remove_frac * X_train.shape[0])

    # result container
    result = {}
    result['loss'] = np.full(total_n_remove + 1, np.nan, dtype=np.float32)

    # trackers
    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    # initial loss
    new_tree = clone(tree).fit(new_X_train, new_y_train)
    res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=f'{0:>5}: {0:>5.2f}%')
    result['loss'][0] = res['loss']

    # initial ranking
    explainer = tree_influence.TreeExplainer(method, params, logger).fit(new_tree, new_X_train, new_y_train)
    inf = explainer.get_local_influence(X_test, y_test).flatten()
    init_ranking = np.argsort(inf)[::-1]

    for n_remove in range(1, total_n_remove + 1):

        # get new training data
        if strategy == 'fixed':
            remove_idxs = init_ranking[:n_remove]
            new_X_train = np.delete(X_train, remove_idxs, axis=0)
            new_y_train = np.delete(y_train, remove_idxs)

        # reestimation influence
        else:
            assert strategy == 'reestimate'

            explainer = tree_influence.TreeExplainer(method, params, logger).fit(new_tree, new_X_train, new_y_train)
            inf = explainer.get_local_influence(X_test, y_test).flatten()
            ranking = np.argsort(inf)[::-1]

            remove_idxs = ranking[:1]
            new_X_train = np.delete(new_X_train, remove_idxs, axis=0)
            new_y_train = np.delete(new_y_train, remove_idxs)

        # data validiity check
        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        # refit new model and measure change in test loss
        else:
            new_tree = clone(tree).fit(new_X_train, new_y_train)

            prefix = f'{n_remove:>5}: {n_remove / X_train.shape[0] * 100:>5.2f}%'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)
            result['loss'][n_remove] = res['loss']

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
    start = time.time()
    tree = tree.fit(X_train, y_train)
    train_time = time.time() - start
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

    # randomly select test instances to compute influence values for
    avail_idxs = np.arange(X_test.shape[0])
    n_test = min(args.n_test, len(avail_idxs))
    test_idxs = select_elements(avail_idxs, rng, n=n_test)

    # shorten remove frac.
    if args.strategy == 'reestimate' and args.n_early_stop > 0:
        args.remove_frac = args.n_early_stop / X_train.shape[0]
        logger.info(f'\nEarly stop active: {args.n_early_stop:,} (remove {args.remove_frac:.3f}%)')

    # estimate experiment time for reestimate strategy
    if args.strategy == 'reestimate' and args.method in ['loo', 'leaf_refit']:
        n_jobs = joblib.cpu_count()
        params['n_jobs'] = 1

        n_remove = int(len(X_train) * args.remove_frac)  # per test
        n_retrains = len(X_train) * n_remove  # per test
        n_sec = train_time * n_retrains  # per test

        logger.info('\nLOO reestimation time estimates:')
        logger.info(f'\tTime to train 1 model: {train_time:.3f}s')

        logger.info(f'\n\tNo. remove (per test): {n_remove:,}')
        logger.info(f'\tNo. retrains (per test): {n_retrains:,}')
        logger.info(f'\tTime (per test): {timedelta(seconds=n_sec)}')

        logger.info(f'\n\tNo. retrains (all): {n_retrains * args.n_test:,}')
        logger.info(f'\tTime (all): {timedelta(seconds=n_sec * args.n_test)}')
        logger.info(f'\t\tWith {n_jobs:,} CPUs: {timedelta(seconds=n_sec * args.n_test / n_jobs)}s')

    # get no. jobs to run in parallel
    if args.n_jobs == -1:
        n_jobs = joblib.cpu_count()

    else:
        assert args.n_jobs >= 1
        n_jobs = min(args.n_jobs, joblib.cpu_count())

    logger.info(f'[INFO] no. jobs: {n_jobs:,}')
    start = time.time()

    with joblib.Parallel(n_jobs=n_jobs) as parallel:

        n_finish = 0
        n_remain = len(test_idxs)

        res_list = []

        while n_remain > 0:
            n = min(min(40, n_jobs), n_remain)

            results = parallel(joblib.delayed(remove_and_reinfluence)
                                             (objective, tree, args.method, params,
                                              X_train, y_train, X_test[[idx]], y_test[[idx]],
                                              args.remove_frac, args.strategy, logger)
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

    # save results
    result['remove_frac'] = np.linspace(0, args.remove_frac, round(args.remove_frac * X_train.shape[0]) + 1)
    result['test_idxs'] = test_idxs
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
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
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)

    # get unique hash for the explainer
    params, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           args.strategy,
                           f'{args.method}_{method_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, params, out_dir)


if __name__ == '__main__':
    main(exp_args.get_reinfluence_args().parse_args())
