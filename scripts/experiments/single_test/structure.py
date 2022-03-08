"""
Compute structural changes.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # tree_influence
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import tree_influence
import util
from config import exp_args


def compute_affinity(tree, X_train, X_test):
    """
    Compute no. times a train example ends in the
    same leaf as the test example.

    Input
        tree: Tree model; must have `apply` method.
        X_train: 2d array of train data.
        X_test: 2d array of test data.

    Return
        1d array of counts of shape=(X_train.shape[0],).
    """
    assert X_train.ndim == X_test.ndim == 2
    assert X_test.shape[0] == 1
    assert hasattr(tree, 'apply')

    train_leaves = tree.apply(X_train)  # shape=(no. train, no. boost, no. class)
    test_leaves = tree.apply(X_test)  # shape=(no. test, no. boost, no. class)

    co_leaf = np.where(train_leaves == test_leaves, 1, 0)
    affinity = np.sum(co_leaf, axis=(1, 2))  # sum over boosts and classes

    return affinity


def compute_weighted_affinity(tree, X_train, X_test):
    """
    Compute no. times a train example ends in the
    same leaf as the test example, weighted by
    1/ni in which ni is the no. examples at leaf i.

    Input
        tree: Tree model; must have `apply` method.
        X_train: 2d array of train data.
        X_test: 2d array of test data.

    Return
        1d array of counts of shape=(X_train.shape[0],).

    Note
        - Make sure `update_node_count` is called before this method.
    """
    assert X_train.ndim == X_test.ndim == 2
    assert X_test.shape[0] == 1
    assert hasattr(tree, 'apply')
    assert hasattr(tree, 'get_leaf_counts')
    assert hasattr(tree, 'get_leaf_weights')
    assert hasattr(tree, 'n_boost_')
    assert hasattr(tree, 'n_class_')

    train_leaves = tree.apply(X_train)  # shape=(no. train, no. boost, no. class)
    test_leaves = tree.apply(X_test)  # shape=(no. test, no. boost, no. class)

    train_weights = get_leaf_weights(tree, train_leaves)

    co_leaf = np.where(train_leaves == test_leaves, 1, 0)  # shape=(no. train, no. boost, no. class)
    affinity = np.sum(co_leaf * train_weights, axis=(1, 2))  # sum over boosts and classes

    return affinity


def get_leaf_weights(model, leaf_idxs):
    """
    Retrieve leaf weights given the leaf indices.

    Input
        model: TreeEnsemble object.
        leaf_idxs: Leaf indices, shape=(no. examples, no. boost, no. class)

    Return
        - 3d array of shape=(no. examples, no. boost, no. class)
    """
    leaf_counts = model.get_leaf_counts()  # shape=(no. boost, no. class)
    leaf_weights = model.get_leaf_weights(-1)  # shape=(total no. leaves,)

    # result container
    weights = np.zeros(leaf_idxs.shape, dtype=util.dtype_t)  # shape=(no. examples, no. boost, no. class)

    n_prev_leaves = 0

    for b_idx in range(model.n_boost_):

        for c_idx in range(model.n_class_):
            leaf_count = leaf_counts[b_idx, c_idx]

            weights[:, b_idx, c_idx] = leaf_weights[n_prev_leaves:][leaf_idxs[:, b_idx, c_idx]]

            n_prev_leaves += leaf_count

    return weights


def remove_and_evaluate(test_idx, objective, ranking, tree,
                        X_train, y_train, X_test, y_test,
                        n_remove_list, logger):

    # only needed for the structure, and a consistent `apply` method between GBDT types
    explainer = tree_influence.TreeExplainer('boostin', {}, logger).fit(tree, X_train, y_train)
    init_affinity = compute_affinity(explainer.model_, X_train, X_test)

    logger.info(f'\nTest index: {test_idx}')

    # result container
    affinity_list = [init_affinity]

    for i, n_remove in enumerate(n_remove_list):
        remove_idxs = ranking[:n_remove]

        new_X_train = np.delete(X_train, remove_idxs, axis=0)
        new_y_train = np.delete(y_train, remove_idxs)
        new_tree = clone(tree).fit(new_X_train, new_y_train)

        new_explainer = tree_influence.TreeExplainer('boostin', {}, logger).fit(new_tree, new_X_train, new_y_train)
        affinity = compute_affinity(new_explainer.model_, X_train, X_test)
        diff = np.abs(init_affinity - affinity)
        diff_single = np.where(diff > 0, 1, 0)

        affinity_list.append(affinity)

        logger.info(f'[{i+1}/{len(n_remove_list)}] n_remove: {n_remove:>5}, '
                    f'affinity sum: {np.sum(affinity):>10,}, '
                    f'diff. sum: {np.sum(diff):>10,}, '
                    f'no. diff > 1: {np.sum(diff_single):>10,}, ')

    result = np.vstack(affinity_list).T  # shape=(no. train, len(n_remove_list))

    return result


def experiment(args, logger, in_dir, out_dir):

    # initialize experiment
    begin = time.time()
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
            n = min(n_jobs, n_remain)

            results = parallel(joblib.delayed(remove_and_evaluate)
                                             (idx, objective, ranking[:, n_finish + i], tree,
                                              X_train, y_train, X_test[[idx]], y_test[[idx]],
                                              args.n_remove, logger)
                                              for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

            # synchronization barrier
            res_list += results

            n_finish += n
            n_remain -= n

            cum_time = time.time() - begin
            logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                        f', cum. time: {cum_time:.3f}s')

    affinity = np.dstack(res_list)  # shape=(no. train, no. checkpoints, no. test)

    # save results
    result['affinity'] = affinity
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for the explainer
    _, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # get input dir., get unique hash for the influence experiment setting
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    in_exp_hash = util.dict_to_hash(exp_dict)

    exp_dir = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{in_exp_hash}',
                           f'{args.method}_{method_hash}')

    # create output dir., get unique hash for the influence experiment setting
    exp_dict['n_remove'] = args.n_remove
    out_exp_hash = util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{out_exp_hash}',
                           f'{args.method}_{method_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, exp_dir, out_dir)


if __name__ == '__main__':
    main(exp_args.get_structure_args().parse_args())
