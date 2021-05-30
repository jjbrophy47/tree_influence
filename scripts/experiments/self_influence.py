"""
Note:
    - For classification, classes MUST be balanced.
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
sys.path.insert(0, here + '/../')
import intent
import util


def evaluate_order(args, order, tree, X_train, y_train, X_test, y_test, logger):

    # pre-removal performance
    res = util.evaluate(args.task, tree, X_test, y_test, logger, prefix='0.00')

    # result container
    result = {}
    result['frac_remove'] = [0]
    result['mse'] = [res['mse']]
    result['acc'] = [res['acc']]
    result['auc'] = [res['auc']]

    # remove train instances
    for frac_remove in np.linspace(0, args.train_frac_to_remove, 10 + 1)[1:]:
        n_remove = int(X_train.shape[0] * frac_remove)
        new_X_train = X_train[order][n_remove:].copy()
        new_y_train = y_train[order][n_remove:].copy()

        if len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        new_tree = clone(tree).fit(new_X_train, new_y_train)
        res = util.evaluate(args.task, new_tree, X_test, y_test, logger, prefix=f'{frac_remove:.2f}')

        # add to results
        result['frac_remove'].append(frac_remove)
        result['mse'].append(res['mse'])
        result['acc'].append(res['acc'])
        result['auc'].append(res['auc'])

    return result


def experiment(args, logger, out_dir):
    begin = time.time()

    # data
    X_train, X_test, y_train, y_test = util.get_toy_data(args.dataset, args.task, args.rs)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}')

    # train tree-ensemble
    tree = util.get_model(args.tree_type, args.task, args.n_estimators, args.max_depth, args.rs)
    tree = tree.fit(X_train, y_train)
    util.evaluate(args.task, tree, X_test, y_test, logger, prefix='Test')

    # order train intances
    start = time.time()

    if args.method == 'random':
        order = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)

    else:
        explainer = intent.TreeExplainer(method=args.method).fit(tree, X_train, y_train)
        self_influence = explainer.get_self_influence()

        if args.task == 'multiclass':
            self_influence = np.abs(self_influence).sum(axis=1)

        order = np.argsort(np.abs(self_influence))[::-1]

    order_time = time.time() - start
    logger.info(f'\norder time: {order_time:.5f}s')

    # remove train instances
    result = evaluate_order(args, order, tree, X_train, y_train, X_test, y_test, logger)

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB
    result['order_time'] = order_time
    result['total_time'] = time.time() - begin
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           f'remove_{args.train_frac_to_remove:.2f}',
                           args.tree_type,
                           f'rs_{args.rs}',
                           args.method)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/self_influence/')

    # Data settings
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--dataset', type=str, default='boston')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)

    # Explainer settings
    parser.add_argument('--method', type=str, default='tracin')

    # Experiment settings
    parser.add_argument('--train_frac_to_remove', type=float, default=0.5)
    parser.add_argument('--rs', type=int, default=1)

    args = parser.parse_args()
    main(args)
