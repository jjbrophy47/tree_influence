"""
Note:
    - For classification, classes MUST be balanced.
"""
import os
import sys
import time
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


def dict_to_hash(d):
    """
    Convert to string and concatenate all values
    in the dict `d` and return the hashed string.
    """
    s = ''
    for k, v in sorted(d.items()):  # alphabetical key sort
        s += str(v)
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def params_to_dict(args):
    """
    Return dict of explainer hyperparameters.
    """
    params = {}

    if args.method == 'tracin':
        params['use_leaf'] = args.use_leaf

    elif args.method == 'leaf_influence':
        params['update_set'] = args.update_set

    elif args.method == 'trex':
        params['kernel'] = args.kernel
        params['target'] = args.target
        params['lmbd'] = args.lmbd
        params['n_epoch'] = args.n_epoch

    # create hash string based on the chosen hyperparameters
    hash_str = dict_to_hash(params)

    params['verbose'] = args.verbose

    return params, hash_str


def evaluate_ranking(args, ranking, tree, X_train, y_train, X_test, y_test, logger):

    # pre-removal performance
    res = util.evaluate(args.objective, tree, X_test, y_test, logger, prefix='0.00')

    # result container
    result = {}
    result['frac_remove'] = [0]
    result['mse'] = [res['mse']]
    result['acc'] = [res['acc']]
    result['auc'] = [res['auc']]

    # remove train instances
    for frac_remove in np.linspace(0, args.train_frac_to_remove, 10 + 1)[1:]:
        n_remove = int(X_train.shape[0] * frac_remove)
        new_X_train = X_train[ranking][n_remove:].copy()
        new_y_train = y_train[ranking][n_remove:].copy()

        if len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        new_tree = clone(tree).fit(new_X_train, new_y_train)
        res = util.evaluate(args.objective, new_tree, X_test, y_test, logger, prefix=f'{frac_remove:.2f}')

        # add to results
        result['frac_remove'].append(frac_remove)
        result['mse'].append(res['mse'])
        result['acc'].append(res['acc'])
        result['auc'].append(res['auc'])

    return result


def experiment(args, logger, params, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)

    # data
    X_train, X_test, y_train, y_test = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}')

    # train tree-ensemble
    tree = util.get_model(args.tree_type, args.objective, args.n_estimators, args.max_depth, args.random_state)
    tree = tree.fit(X_train, y_train)
    util.evaluate(args.objective, tree, X_test, y_test, logger, prefix='Test')

    # rank train intances
    start = time.time()

    if args.method == 'random':
        ranking = rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)

    else:
        explainer = intent.TreeExplainer(method=args.method, params=params).fit(tree, X_train, y_train)
        global_influence = explainer.get_global_influence()

        # sort train examples based on largest absolute influence
        ranking = np.argsort(np.abs(global_influence))[::-1]
        # ranking = np.argsort(np.abs(global_influence))
        # ranking = np.argsort(global_influence)[::-1]

        print(ranking)
        print(global_influence[ranking])
        print(y_train[ranking])

    rank_time = time.time() - start
    logger.info(f'\nrank time: {rank_time:.5f}s')

    # remove train instances
    result = evaluate_ranking(args, ranking, tree, X_train, y_train, X_test, y_test, logger)

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB
    result['rank_time'] = rank_time
    result['total_time'] = time.time() - begin
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get method params and unique settings hash
    params, hash_str = params_to_dict(args)

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           f'remove_{args.train_frac_to_remove:.2f}',
                           args.tree_type,
                           f'rs_{args.random_state}',
                           f'{args.method}_{hash_str}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, params, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/global_influence/')

    # Data settings
    parser.add_argument('--objective', type=str, default='regression')
    parser.add_argument('--dataset', type=str, default='synthetic_regression')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)

    # Explainer settings
    parser.add_argument('--method', type=str, default='random')
    parser.add_argument('--use_leaf', type=int, default=1)  # TracIn
    parser.add_argument('--update_set', type=int, default=-1)  # LeafInfluence
    parser.add_argument('--kernel', type=str, default='lpw')  # Trex
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex
    parser.add_argument('--verbose', type=int, default=0)  # TracIn, LeafInfluence, + Trex

    # Experiment settings
    parser.add_argument('--train_frac_to_remove', type=float, default=0.5)
    parser.add_argument('--random_state', type=int, default=1)

    args = parser.parse_args()
    main(args)
