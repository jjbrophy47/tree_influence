"""
Compute influence for a validation set of examples.
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
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # tree_influence
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import tree_influence
import util
from config import exp_args
from single_test.influence import get_special_case_tol


def experiment(args, logger, params, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # use a fraction of the test data for validation
    stratify = None if objective == 'regression' else y_test
    test_indices = np.arange(y_test.shape[0])
    X_val, X_test, y_val, y_test, val_idxs, test_idxs = train_test_split(X_test, y_test, test_indices,
                                                                         test_size=1.0 - args.val_frac,
                                                                         stratify=stratify,
                                                                         random_state=args.random_state)

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. val.: {X_val.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)

    tree = tree.fit(X_train, y_train)
    res_clean = util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test (clean)')

    # fit explainer
    start = time.time()
    explainer = tree_influence.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s\n')

    # compute influence
    start2 = time.time()

    # aggregate local influences
    local_influence = explainer.get_local_influence(X_val, y_val)  # shape=(no. test,)
    influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

    inf_time = time.time() - start2
    logger.info(f'[INFO] influence time: {inf_time:.5f}s\n')

    total_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {total_time:.3f}s')

    # save results
    result['influence'] = influence
    result['val_idxs'] = val_idxs
    result['test_idxs'] = test_idxs
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = total_time
    result['tree_params'] = tree.get_params()

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'val_frac': args.val_frac}
    exp_hash = util.dict_to_hash(exp_dict)

    # get unique hash for the explainer
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)
    params, hash_str = util.explainer_params_to_dict(args.method, vars(args))

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

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_influence_set_args().parse_args())
