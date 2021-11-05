"""
Compute global influence.
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


def experiment(args, logger, params, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, X_val, y_val, objective = util.get_data(args.data_dir, args.dataset,
                                                                              adaptation=True)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. val: {X_val.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s\n')

    # compute influence
    start2 = time.time()

    # aggregate local influences
    if args.strategy == 'test_sum':
        local_influence = explainer.get_local_influence(X_val, y_val)  # shape=(no. test,)
        influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

    elif args.strategy == 'test_mean':
        local_influence = explainer.get_local_influence(X_val, y_val)  # shape=(no. test,)
        influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

    else:
        assert args.strategy == 'batch'
        raise ValueError('Not implemented')

    inf_time = time.time() - start2
    logger.info(f'\n[INFO] inf. time: {inf_time:.3f}s')

    # remove example with net negative influence on the test set
    neg_idxs = np.where(influence < 0)[0]
    new_X_train = np.delete(X_train, neg_idxs, axis=0)
    new_y_train = np.delete(y_train, neg_idxs)
    logger.info(f'\n[INFO] removing {len(neg_idxs):,} examples...')

    # re-weight positive examples relative to influence
    pos_idxs = np.where(influence >= 0)[0]
    sample_weight = influence[pos_idxs] / np.max(influence[pos_idxs])
    logger.info(f'[INFO] re-weighting pos. influence examples...\n')

    new_tree = clone(tree).fit(new_X_train, new_y_train, sample_weight=sample_weight)
    res = util.eval_pred(objective, new_tree, X_test, y_test, logger, prefix='Test (w/ reweight)')

    new_tree = clone(tree).fit(new_X_train, new_y_train)
    res = util.eval_pred(objective, new_tree, X_test, y_test, logger, prefix='Test (no reweight)')

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['influence'] = influence
    result['loss'] = res['loss']
    result['acc'] = res['acc']
    result['auc'] = res['auc']
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'strategy': args.strategy}
    exp_hash = util.dict_to_hash(exp_dict)

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
                           f'exp_{exp_hash}',
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
    parser.add_argument('--out_dir', type=str, default='output/adaptation/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--strategy', type=str, default='test_sum')

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

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    args = parser.parse_args()
    main(args)
