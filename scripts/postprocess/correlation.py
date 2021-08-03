"""
Measure correlation between two influence methods.
"""
import os
import sys
import time
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util


def experiment(args, logger, in_dir1, in_dir2, out_dir):

    # initialize experiment
    begin = time.time()

    # get influence results
    inf_res1 = np.load(os.path.join(in_dir1, 'results.npy'), allow_pickle=True)[()]
    inf_res2 = np.load(os.path.join(in_dir2, 'results.npy'), allow_pickle=True)[()]

    # evaluate influence ranking
    start = time.time()
    result = {}

    inf1 = inf_res1['influence']
    inf2 = inf_res2['influence']

    # average influence values over all test examples
    if args.inf_obj == 'local':

        # sort influence values
        for i in range(inf1.shape[1]):
            idxs = np.argsort(inf1[:, i])[::-1]
            inf1[:, i] = inf1[:, i][idxs]
            inf2[:, i] = inf2[:, i][idxs]

        inf1 = inf1.mean(axis=1)
        inf2 = inf2.mean(axis=1)

    # compute correlation for the entire length of influence values
    fig, axs = plt.subplots(1, len(args.zoom), figsize=(4 * len(args.zoom), 4), sharey=True)

    for i, zoom in enumerate(args.zoom):
        assert zoom > 0 and zoom <= 1.0

        n = int(len(inf1) * zoom)
        i1 = inf1[:n]
        i2 = inf2[:n]

        # shape=(no. train,)
        pearson = pearsonr(i1, i2)[0]
        spearman = spearmanr(i1, i2)[0]
        r2score = r2_score(i1, i2)

        ax = axs[i]
        ax.scatter(i1, i2, label=f'p: {pearson:.3f}\ns: {spearman:.3f}\nr^2: {r2score:.3f}')
        ax.set_title(f'first {zoom * 100}% (sorted by {args.method1})')
        ax.set_xlabel(args.method1)
        ax.legend(fontsize=6)
        if i == 0:
            ax.set_ylabel(args.method2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.method1}_{args.method2}.png'), bbox_inches='tight')
    plt.show()


def main(args):

    # get method params and unique settings hash
    _, hash_str1 = util.explainer_params_to_dict(args.method1, vars(args))
    _, hash_str2 = util.explainer_params_to_dict(args.method2, vars(args))

    # method1 dir
    in_dir1 = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'rs_{args.random_state}',
                           args.inf_obj,
                           f'{args.method1}_{hash_str1}')

    in_dir2 = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'rs_{args.random_state}',
                           args.inf_obj,
                           f'{args.method2}_{hash_str2}')

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.inf_obj,
                           args.dataset)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, in_dir1, in_dir2, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='output/influence/')
    parser.add_argument('--out_dir', type=str, default='output/plot/correlation/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='synthetic_regression')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Explainer settings
    parser.add_argument('--use_leaf', type=int, default=1)  # BoostIn
    parser.add_argument('--local_op', type=str, default='normal')  # BoostIn

    parser.add_argument('--update_set', type=int, default=0)  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default='dot_prod')  # Similarity

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex & Similarity
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--global_op', type=str, default='self')  # TREX, LOO, and DShap

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random

    # Experiment settings
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--method1', type=str, default='random')
    parser.add_argument('--method2', type=str, default='boostin')
    parser.add_argument('--zoom', type=float, nargs='+', default=[0.01, 0.1, 0.5, 1.0])

    args = parser.parse_args()
    main(args)
