"""
Explain a test example prediction.
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
import seaborn as sns
import matplotlib.pyplot as plt
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
    X_train, X_test, y_train, y_test, feature, objective = util.get_data(args.data_dir, args.dataset, feature=True)

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # get potentially quetionable test examples
    df = pd.DataFrame(X_test, columns=feature)
    df['score_factor'] = y_test
    qf = df[df['score_factor'] == 1]  # COMPAS prediction
    qf = qf[qf['African_American'] == 1]
    qf = qf[qf['Number_of_Priors'] == 1]
    qf = qf[qf['Two_yr_Recidivism'] == 0]
    qf = qf[qf['Misdemeanor'] == 0]
    qf = qf[qf['Age_Below_TwentyFive'] == 1]
    qf = qf[qf['Female'] == 0]
    explain_idxs = qf.index.values

    # fit model
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test (clean)')

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_
    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # select a test example to explain
    pred = tree.predict(X_test)
    proba = tree.predict_proba(X_test)[:, 1]
    correct_idxs = np.where(pred == y_test)[0]
    avail_idxs = np.intersect1d(explain_idxs, correct_idxs)
    test_idx = rng.choice(avail_idxs, size=1)

    # compute influence
    start2 = time.time()
    influence = explainer.get_local_influence(X_test[(test_idx,)], y_test[(test_idx,)])[:, 0]
    inf_time = time.time() - start2
    logger.info(f'[INFO] influence time: {inf_time:.5f}s')

    # show test example
    test_df = pd.DataFrame(X_test[(test_idx,)], columns=feature)
    test_df['label'] = y_test[(test_idx,)]
    test_df['conf'] = proba[(test_idx,)]
    logger.info(f'\nTest example:\n{test_df}')

    # show top-k examples that reduce loss of the test example
    # topk_idxs = np.argsort(influence)[::-1][:args.topk]
    topk_idxs = np.argsort(influence)[:args.topk]

    train_df = pd.DataFrame(X_train[topk_idxs], columns=feature)
    train_df['label'] = y_train[topk_idxs]
    train_df['inf'] = influence[topk_idxs]
    logger.info(f'\nTop Train examples:\n{train_df}')

    # sns.histplot(influence)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(influence)), influence[np.argsort(influence)[::-1]])
    # ax.set_title(f'Recidivism Risk: High (confidence: {proba[test_idx][0] * 100:.1f}%);'
    #              f'\nAfrican American, female, no prev. recidivism, no priors, '
    #              f'age<25, no misdemeanors', fontsize=6)
    # ax.set_title(f'Recidivism Risk: High (confidence: {proba[test_idx][0] * 100:.1f}%);'
    #              f'\nAfrican American, male, no prev. recidivism, 1 prior, '
    #              f'no misdemeanors, age<25', fontsize=6)
    ax.set_title(f'Recidivism Risk: Low (confidence: {100 - proba[test_idx][0] * 100:.1f}%);'
                 f'\nCaucasian, male, no prev. recidivism, 1 prior, '
                 f'no misdemeanors, age<25', fontsize=6)
    ax.set_ylabel('Influence value')
    ax.set_xlabel('Train index (sorted by influence)')
    plt.savefig(os.path.join(out_dir, 'plot.pdf'))
    plt.show()

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
                           f'{args.method}_{hash_str}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, params, out_dir)

    util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/explain/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--topk', type=int, default=10)

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

    args = parser.parse_args()
    main(args)
