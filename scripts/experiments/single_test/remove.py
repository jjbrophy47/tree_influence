"""
Remove train data and measure loss.
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
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import util
from config import exp_args
from influence import get_special_case_tol
from influence import select_n_jobs


def remove_and_evaluate(objective, ranking, tree,
                        X_train, y_train, X_test, y_test,
                        remove_frac_list, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_loss

    # result container
    result = {}
    result['loss'] = np.full(len(remove_frac_list), np.nan, dtype=np.float32)
    result['pred_label'] = np.full(len(remove_frac_list), np.nan, dtype=np.float32)

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
            result['pred_label'][i] = res['pred_label']

    return result


def experiment(args, logger, in_dir, out_dir):

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

    # read influence values
    inf_res = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]
    influence = inf_res['influence']  # shape=(no. train, no. test)
    test_idxs = inf_res['test_idxs']  # shape=(no. test,)

    # get ranking
    ranking = np.argsort(influence, axis=0)[::-1]  # shape=(no. train, no. test)

    # get no. jobs to run in parallel
    n_jobs = select_n_jobs(args.n_jobs)
    logger.info(f'\n[INFO] no. jobs: {n_jobs:,}')

    with joblib.Parallel(n_jobs=n_jobs) as parallel:

        n_finish = 0
        n_remain = len(test_idxs)

        res_list = []

        while n_remain > 0:
            n = min(n_jobs, n_remain)

            results = parallel(joblib.delayed(remove_and_evaluate)
                                             (objective, ranking[:, n_finish + i], tree,
                                              X_train, y_train, X_test[[idx]], y_test[[idx]],
                                              args.remove_frac,
                                              logger) for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

            # synchronization barrier
            res_list += results

            n_finish += n
            n_remain -= n

            cum_time = time.time() - begin
            logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                        f', cum. time: {cum_time:.3f}s')

    # save results
    result['remove_frac'] = np.array(args.remove_frac, dtype=np.float32)
    result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['pred_label'] = np.vstack([res['pred_label'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for the explainer
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)
    params, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # get input dir., get unique hash for the influence experiment setting
    exp_dict = {'n_test': args.n_test}
    exp_hash = util.dict_to_hash(exp_dict)

    in_dir = os.path.join(args.in_dir,
                          args.dataset,
                          args.tree_type,
                          f'exp_{exp_hash}',
                          f'{args.method}_{method_hash}')

    # create output dir., get unique hash for the influence experiment setting
    exp_dict['remove_frac'] = args.remove_frac
    exp_hash = util.dict_to_hash(exp_dict)

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

    experiment(args, logger, in_dir, out_dir)


if __name__ == '__main__':
    main(exp_args.get_remove_args().parse_args())
