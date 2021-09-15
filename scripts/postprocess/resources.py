"""
Aggregate results for a single dataset.
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
import util
from experiments import util as exp_util
from config import post_args


def process(args, out_dir, logger):

    color, line, label = util.get_plot_dicts()

    results = []
    r1 = {}

    for random_state in range(1, args.n_repeat + 1):
        logger.info(f'\nRandom state: {random_state}')

        r1[random_state] = {}

        exp_dir = os.path.join(args.in_dir,
                               args.dataset,
                               args.tree_type,
                               f'random_state_{random_state}')
        res_list = util.get_results(args, args.in_dir, exp_dir, logger)
        res_list = util.filter_results(res_list, args.skip)

        for method, d in res_list:
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
    X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, args.dataset)

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
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_resources_args().parse_args())
