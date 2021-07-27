"""
Evaluate influence value ranking via remove and retrain.
"""
import os
import sys
import time
import joblib
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import intent
import util


def remove_and_retrain(inf_obj, objective, ranking, tree, X_train, y_train,
                       X_test, y_test, remove_frac, n_ckpt, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_pred if inf_obj == 'global' else util.eval_loss

    # pre-removal performance
    res = eval_fn(objective, tree, X_test, y_test, logger, prefix=f'{0:>5}: {0:>5.2f}%')

    # get list of remove fractions
    remove_frac_arr = np.linspace(0, remove_frac, n_ckpt + 1)[1:]

    # result container
    result = {}
    result['remove_frac'] = np.concatenate([np.array([0.0]), remove_frac_arr])
    for key in res.keys():
        result[key] = np.full(remove_frac_arr.shape[0] + 1, np.nan, dtype=np.float32)
        result[key][0] = res[key]

    # remove train instances
    for i, remove_frac in enumerate(remove_frac_arr):
        n_remove = int(X_train.shape[0] * remove_frac)

        new_X_train = X_train[ranking][n_remove:].copy()
        new_y_train = y_train[ranking][n_remove:].copy()

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            new_tree = clone(tree).fit(new_X_train, new_y_train)
            prefix = f'{i + 1:>5}: {remove_frac * 100:>5.2f}%'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)

            # add to results
            result['remove_frac'][i + 1] = remove_frac
            for key in res.keys():
                result[key][i + 1] = res[key]

    return result


def experiment(args, logger, params, in_dir, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)

    # get influence results
    inf_res = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**inf_res['tree_params'])
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')
    logger.info('')

    # evaluate influence ranking
    start = time.time()
    result = {}

    influence = inf_res['influence']

    # sort influence starting from most pos. to most neg.
    if args.inf_obj == 'global':
        ranking = np.argsort(influence)[::-1]
        res = remove_and_retrain(args.inf_obj, objective, ranking, tree, X_train, y_train,
                                 X_test, y_test, args.remove_frac, args.n_ckpt, logger)
        result.update(res)  # add ROAR results to result object

    else:
        assert args.inf_obj == 'local', 'only local supported!'

        test_idxs = inf_res['test_idxs']

        # get no. jobs to run in parallel
        if args.n_jobs == -1:
            n_jobs = joblib.cpu_count()

        else:
            assert args.n_jobs >= 1
            n_jobs = min(args.n_jobs, joblib.cpu_count())

        logger.info(f'[INFO] no. jobs: {n_jobs:,}')

        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            n_finish = 0
            n_remain = len(test_idxs)

            res_list = []

            while n_remain > 0:
                n = min(min(10, n_jobs), n_remain)

                results = parallel(joblib.delayed(remove_and_retrain)
                                                 (args.inf_obj, objective,
                                                  np.argsort(influence[:, n_finish + i])[::-1],
                                                  tree, X_train, y_train, X_test[[idx]], y_test[[idx]],
                                                  args.remove_frac, args.n_ckpt, logger)
                                                  for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

                # synchronization barrier
                res_list += results

                n_finish += n
                n_remain -= n

                cum_time = time.time() - start
                logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                            f', cum. time: {cum_time:.3f}s')

            # combine results from each test example
            for key in res_list[0].keys():
                result[key] = np.vstack([r[key] for r in res_list])  # shape=(no. test, no. completed ckpts)

    roar_time = time.time() - start
    logger.info(f'ROAR time: {roar_time:.5f}s')

    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin

    # save results
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get method params and unique settings hash
    params, hash_str = util.explainer_params_to_dict(args.method, vars(args))

    # influence dir
    in_dir = os.path.join(args.in_dir,
                          args.dataset,
                          args.tree_type,
                          f'rs_{args.random_state}',
                          args.inf_obj,
                          f'{args.method}_{hash_str}')

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'rs_{args.random_state}',
                           args.inf_obj,
                           f'{args.method}_{hash_str}')

    # exit if results already exist
    if args.skip and os.path.exists(os.path.join(out_dir, 'results.npy')):
        fp = os.path.exists(os.path.join(out_dir, 'results.npy'))
        print(f'results present, skipping {fp}...')
        exit(0)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, params, in_dir, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='output/influence/')
    parser.add_argument('--out_dir', type=str, default='output/roar/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='synth_regression')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Explainer settings
    parser.add_argument('--method', type=str, default='random')

    parser.add_argument('--use_leaf', type=int, default=1)  # BoostIn
    parser.add_argument('--local_op', type=str, default='normal')  # BoostIn

    parser.add_argument('--update_set', type=int, default=-1)  # LeafInfluence

    parser.add_argument('--similarity', type=str, default='dot_prod')  # Similarity

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex & Similarity
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    # Experiment settings
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--inf_obj', type=str, default='global')
    parser.add_argument('--remove_frac', type=float, default=0.5)
    parser.add_argument('--n_ckpt', type=int, default=200)
    parser.add_argument('--n_jobs', type=int, default=-1)

    args = parser.parse_args()
    main(args)
