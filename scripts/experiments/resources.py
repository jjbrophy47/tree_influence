"""
Measure resource usage.
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
import intent
import util


def experiment(args, logger, params, random_state, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # fit model
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test (clean)')

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_
    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # select a test example to explain
    test_idx = rng.choice(np.arange(len(X_test)), size=1)

    # compute influence
    start2 = time.time()
    local_influence = explainer.get_local_influence(X_test[(test_idx,)], y_test[(test_idx,)])
    inf_time = time.time() - start2
    logger.info(f'[INFO] influence time: {inf_time:.5f}s')

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):
    assert args.n_jobs == 1

    for random_state in range(1, args.n_repeat + 1):

        # select seed
        if args.seed > 0:
            assert args.n_repeat == 1
            seed = args.seed

        else:
            seed = random_state

        # get unique hash for the explainer
        params, hash_str = util.explainer_params_to_dict(args.method, vars(args))

        # special cases
        if args.method == 'leaf_influence':
            if args.dataset == 'flight_delays':
                params['atol'] = 1e-1

        # create output dir
        out_dir = os.path.join(args.out_dir,
                               args.dataset,
                               args.tree_type,
                               f'random_state_{seed}',
                               f'{args.method}_{hash_str}')

        # create output directory and clear previous contents
        os.makedirs(out_dir, exist_ok=True)
        util.clear_dir(out_dir)

        logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
        logger.info(args)
        logger.info(f'\ntimestamp: {datetime.now()}')

        experiment(args, logger, params, seed, out_dir)

        util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/resources/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Explainer settings
    parser.add_argument('--method', type=str, default='random')

    parser.add_argument('--leaf_scale', type=float, default=-1.0)  # BoostIn
    parser.add_argument('--local_op', type=str, default='normal')  # BoostIn

    parser.add_argument('--update_set', type=int, default=0)  # LeafInfluence

    parser.add_argument('--similarity', type=str, default='dot_prod')  # Similarity

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex & similarity
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--sub_frac', type=float, default=0.7)  # SubSample
    parser.add_argument('--n_iter', type=int, default=4000)  # SubSample

    parser.add_argument('--n_jobs', type=int, default=1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    # Additional settings
    parser.add_argument('--n_repeat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()
    main(args)
