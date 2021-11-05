"""
Train multiple models to estimate prediction variance.
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
from ngboost import NGBRegressor

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import intent
import util
from influence import select_elements


def train_and_predict(objective, tree, rng, sub_frac,
                      X_train, y_train, X_test, y_test, logger):

    n = int(X_train.shape[0] * sub_frac)
    idxs = rng.choice(X_train.shape[0], size=n, replace=False)

    new_X_train = X_train[idxs].copy()
    new_y_train = y_train[idxs].copy()

    new_tree = clone(tree).fit(new_X_train, new_y_train)

    # store ALL train and test predictions
    if objective == 'regression':
        y_train_pred = new_tree.predict(X_train).reshape(-1, 1)  # shape=(no. train, 1)
        y_test_pred = new_tree.predict(X_test).reshape(-1, 1)  # shape=(no. test, 1)

    elif objective == 'binary':
        y_train_pred = new_tree.predict_proba(X_train)[:, 1].reshape(-1, 1)  # shape=(no. train, 1)
        y_test_pred = new_tree.predict_proba(X_test)[:, 1].reshape(-1, 1)  # shape=(no. test, 1)

    else:
        assert objective == 'multiclass'
        y_train_pred = new_tree.predict_proba(X_train)  # shape=(no. train, no. class)
        y_test_pred = new_tree.predict_proba(X_test)  # shape=(no. test, no. class)

    return y_train_pred, y_test_pred


def experiment(args, logger, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    assert objective == 'regression', 'Regression objective only!'

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

    # get no. jobs to run in parallel
    if args.n_jobs == -1:
        n_jobs = joblib.cpu_count()

    else:
        assert args.n_jobs >= 1
        n_jobs = min(args.n_jobs, joblib.cpu_count())

    if args.method == 'ensemble':

        logger.info(f'\n[INFO] no. jobs: {n_jobs:,}')

        start = time.time()

        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            n_finish = 0
            n_remain = args.n_ensemble

            y_train_pred_list = []
            y_test_pred_list = []

            while n_remain > 0:
                n = min(min(5, n_jobs), n_remain)

                results = parallel(joblib.delayed(train_and_predict)
                                                 (objective, tree, rng, args.sub_frac,
                                                  X_train, y_train, X_test, y_test, logger)
                                                  for i in range(n))

                # synchronization barrier
                for y_train_pred, y_test_pred in results:
                    y_train_pred_list.append(y_train_pred)
                    y_test_pred_list.append(y_test_pred)

                n_finish += n
                n_remain -= n

                cum_time = time.time() - start
                logger.info(f'[INFO] ensembles finished: {n_finish:,} / {args.n_ensemble:,}'
                            f', cum. time: {cum_time:.3f}s')

            # combine results from each test example
            y_train_pred = np.hstack(y_train_pred_list)  # shape=(no. train, no. ensemble)
            y_test_pred = np.hstack(y_test_pred_list)  # shape=(no. test, no. ensemble)

            y_train_pred_mean = np.mean(y_train_pred, axis=1)
            y_train_pred_scale = np.var(y_train_pred, axis=1)  # variance

            y_test_pred_mean = np.mean(y_test_pred, axis=1)
            y_test_pred_scale = np.var(y_test_pred, axis=1)

    else:
        assert args.method == 'ngboost'
        
        tree = NGBRegressor().fit(X_train, y_train)
        y_train_dists = tree.pred_dist(X_train)
        y_test_dists = tree.pred_dist(X_test)

        y_train_pred_mean = y_train_dists.params['loc']
        y_train_pred_scale = y_train_dists.params['scale'] ** 2  # variance

        y_test_pred_mean = y_test_dists.params['loc']
        y_test_pred_scale = y_test_dists.params['scale'] ** 2

    # save results
    result['test_idxs'] = test_idxs
    result['y_train_pred_mean'] = y_train_pred_mean
    result['y_train_pred_scale'] = y_train_pred_scale
    result['y_test_pred_mean'] = y_test_pred_mean
    result['y_test_pred_scale'] = y_test_pred_scale
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # TEMP: rough analysis
    import matplotlib.pyplot as plt

    targets = y_test[test_idxs]
    preds = y_test_pred_mean[test_idxs]
    var = y_test_pred_scale[test_idxs]

    title = f'{args.dataset.capitalize()}'
    if args.method == 'ensemble':
        title += f' ({args.n_ensemble:,} models)'

    fig, ax = plt.subplots()
    ax.errorbar(targets, preds, yerr=var, fmt='.', capsize=2, color='k', elinewidth=1)
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title(title)

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.savefig(os.path.join(out_dir, 'plot.png'), bbox_inches='tight')


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'n_test': args.n_test}
    exp_hash = util.dict_to_hash(exp_dict)

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           args.method)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/variance/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--n_test', type=int, default=100)

    # Method settings
    parser.add_argument('--method', type=str, default='ensemble')
    parser.add_argument('--n_ensemble', type=int, default=10)
    parser.add_argument('--sub_frac', type=float, default=0.7)

    # Additional Settings
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)

    args = parser.parse_args()
    main(args)
