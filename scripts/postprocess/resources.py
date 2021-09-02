"""
Aggregate results.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util as pp_util
from experiments import util
from leaf_analysis import filter_results


def process(args, out_dir, logger):

    color, line, label = pp_util.get_plot_dicts()

    results = []
    r1 = {}

    for random_state in args.random_state:
        logger.info(f'\nRandom state: {random_state}')

        r1[random_state] = {}

        exp_dir = os.path.join(args.in_dir,
                               args.dataset,
                               args.tree_type,
                               f'random_state_{random_state}')
        res = filter_results(pp_util.get_results(args, args.in_dir, logger,
                                                 exp_hash='', temp_dir=exp_dir), args.skip)

        for method, d in res:
            r = {}
            r['mem_GB'] = d['max_rss_MB']  # results were run on Linux
            r['fit_time'] = d['fit_time']
            r['inf_time'] = d['inf_time']
            r['total_time'] = r['fit_time'] + r['inf_time']

            r1[random_state][label[method]] = r

    # average over random states
    for method in r1[random_state].keys():
        r = {'method': method}

        for metric in r1[random_state][method].keys():

            metric_vals = [r1[rs][method][metric] for rs in r1.keys()]
            metric_mean = np.mean(metric_vals)
            metric_std = np.std(metric_vals)

            r[f'{metric}_mean'] = metric_mean
            r[f'{metric}_std'] = metric_std
            r['n'] = len(metric_vals)

        results.append(r)

    df = pd.DataFrame(results)
    logger.info(f'\nResults:\n{df}')

    # get dataset
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    ax = axs[0]
    ax.errorbar(df['method'], df['total_time_mean'], yerr=df['total_time_std'],
                linestyle='', marker='o', capsize=2.5, color='k')
    ax.set_yscale('log')
    ax.set_ylabel('Total time (s)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax = axs[1]
    ax.errorbar(df['method'], df['mem_GB_mean'], yerr=df['mem_GB_std'],
                linestyle='', marker='o', capsize=2.5, color='k')
    ax.set_ylabel('Memory (GB)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt_dir = os.path.join(args.out_dir,
                           args.tree_type)
    os.makedirs(plt_dir, exist_ok=True)
    fp = os.path.join(plt_dir, f'{args.dataset}')

    plt.tight_layout()
    plt.savefig(fp + '.png', bbox_inches='tight')
    plt.show()


def main(args):

    out_dir = os.path.join(args.out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_resources/')
    parser.add_argument('--out_dir', type=str, default='output/plot/resources/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--strategy', type=str, nargs='+', default=['test_sum', 'self'])
    parser.add_argument('--noise', type=str, default='target')
    parser.add_argument('--noise_frac', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--check_frac', type=float, default=0.1)

    # additional settings
    parser.add_argument('--random_state', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['trex', 'similarity', 'boostin2', 'leaf_influenceSP', 'subsample', 'loo'])
    parser.add_argument('--skip', type=str, nargs='+', default=[])
    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1, 0])  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default=['dot_prod'])  # Similarity

    parser.add_argument('--kernel', type=str, nargs='+', default=['lpw'])  # Trex & Similarity
    parser.add_argument('--target', type=str, nargs='+', default=['actual'])  # Trex
    parser.add_argument('--lmbd', type=float, nargs='+', default=[0.003])  # Trex
    parser.add_argument('--n_epoch', type=str, nargs='+', default=[3000])  # Trex

    parser.add_argument('--trunc_frac', type=float, nargs='+', default=[0.25])  # DShap
    parser.add_argument('--check_every', type=int, nargs='+', default=[100])  # DShap

    parser.add_argument('--sub_frac', type=float, nargs='+', default=[0.7])  # SubSample
    parser.add_argument('--n_iter', type=int, nargs='+', default=[4000])  # SubSample

    parser.add_argument('--global_op', type=str, nargs='+', default=['self', 'expected'])  # TREX, LOO, DShap

    args = parser.parse_args()
    main(args)
