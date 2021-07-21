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
        inf1 = inf1.mean(axis=1)
        inf2 = inf2.mean(axis=1)

    # compute correlation for the entire length of influence values
    if args.n_sample == 1:

        if args.frac > 0.0 and args.frac < 1.0:
            n = int(len(inf1) * args.frac)
            inf1 = inf1[:n]
            inf2 = inf2[:n]

        # shape=(no. train,)
        pearson = pearsonr(inf1, inf2)[0]
        spearman = spearmanr(inf1, inf2)[0]
        r2score = r2_score(inf1, inf2)

        fig, ax = plt.subplots()
        label = f'p: {pearson:.3f}\ns: {spearman:.3f}\nr^2: {r2score:.3f}'
        ax.scatter(inf1, inf2, label=label)
        ax.set_title(f'{args.dataset}, {args.inf_obj}')
        ax.set_xlabel(args.method1)
        ax.set_ylabel(args.method2)
        ax.legend(fontsize=6)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{args.method1}_{args.method2}.png'), bbox_inches='tight')
        plt.show()

    # compute correlation for increasing lengths of the influence values
    elif args.n_sample > 1:
        frac_list = np.linspace(0, 1.0, args.n_sample + 1)[1:]

        frac_arr = np.zeros(len(frac_list))
        p_arr = np.zeros(len(frac_list))
        s_arr = np.zeros(len(frac_list))
        r2_arr = np.zeros(len(frac_list))

        for i, frac in enumerate(frac_list):
            n = int(len(inf1) * frac)

            inf_a = inf1[:n]
            inf_b = inf2[:n]

            p_arr[i] = pearsonr(inf_a, inf_b)[0]
            s_arr[i] = spearmanr(inf_a, inf_b)[0]
            r2_arr[i] = r2_score(inf_a, inf_b)
            frac_arr[i] = frac

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for i, (arr, name) in enumerate([(p_arr, 'pearson'), (s_arr, 'spearman'), (r2_arr, 'r2')]):
            ax = axs[i]
            ax.plot(frac_arr * 100, arr)
            ax.set_xlabel('% influence values')
            ax.set_ylabel(name)

        plt.tight_layout()
        plt.show()


def main(args):

    # get method params and unique settings hash
    _, hash_str1 = util.explainer_params_to_dict(args.method1, vars(args))
    _, hash_str2 = util.explainer_params_to_dict(args.method2, vars(args))

    # get str for influence objective
    inf_type = 'global'
    if args.inf_obj == 'local':
        inf_type = f'local_{args.test_select}'

    # method1 dir
    in_dir1 = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'rs_{args.random_state}',
                           inf_type,
                           f'{args.method1}_{hash_str1}')

    in_dir2 = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'rs_{args.random_state}',
                           inf_type,
                           f'{args.method2}_{hash_str2}')

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           inf_type,
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
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)

    # Explainer settings
    parser.add_argument('--use_leaf', type=int, default=1)  # BoostIn

    parser.add_argument('--update_set', type=int, default=-1)  # LeafInfluence

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex
    parser.add_argument('--use_alpha', type=int, default=0)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--global_op', type=str, default='self')  # TREX, LOO, and DShap

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--verbose', type=int, default=1)  # BoostIn, LeafInfluence, Trex, LOO, DShap

    # Experiment settings
    parser.add_argument('--inf_obj', type=str, default='global')
    parser.add_argument('--test_select', type=str, default='random')  # local
    parser.add_argument('--method1', type=str, default='random')
    parser.add_argument('--method2', type=str, default='boostin')
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--n_sample', type=int, default=1)

    args = parser.parse_args()
    main(args)
