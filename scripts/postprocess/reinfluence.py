"""
Plot results for a single dataset.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util
from config import post_args


def process(args, out_dir, logger):

    n_test = None
    results = []

    # get dataset
    X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, args.dataset)

    # get results
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    for strategy in args.strategy:

        exp_dir = os.path.join(args.in_dir,
                               args.dataset,
                               args.tree_type,
                               f'exp_{exp_hash}',
                               strategy)

        res_list = util.get_results(args, exp_dir, logger)
        res_list = util.filter_results(res_list, args.skip)

        for method, res in res_list:
            new_res = {'remove_frac': res['remove_frac'], 'loss': res['loss']}
            results.append((method, strategy, new_res))

    # plot
    util.plot_settings(fontsize=17)

    color, line, label = util.get_plot_dicts()
    marker = {'random_': 'o', 'loo_': 'd', 'boostin_': '*', 'leaf_refit_6bb61e3b7bce0931da574d19d1d82c88': '+'}

    fig, ax = plt.subplots()

    for i, (method, strategy, res) in enumerate(results):

        if method == 'random_' and strategy == 'reestimate':
            continue

        # sanity check
        if i == 0:
            n_test = res['loss'].shape[0]

        else:
            temp = res['loss'].shape[0]
            assert n_test == temp, f'Inconsistent no. test: {temp:,} != {n_test:,}'

        # plot loss
        x = res['remove_frac'] * 100
        y = res['loss'].mean(axis=0)
        y_err = sem(res['loss'], axis=0)
        y_err = y_err if args.std_err else None

        linestyle = '-' if strategy == 'fixed' else '--'
        temp_label = f'{label[method]}' if strategy == 'fixed' else ''

        ax.errorbar(x, y, yerr=y_err, label=temp_label, linestyle=linestyle,
                    color=color[method], marker=marker[method],
                    alpha=0.75, markevery=args.markevery)

        # add rectangle around the first deletion of LOO
        if method == 'loo_' and strategy == 'fixed':
            width = x[2] - x[0]
            height = y[1] - y[0]
            rect = patches.Rectangle((x[0], y[0]), width, height,
                                     linewidth=1, edgecolor='gray', facecolor='none',
                                     capstyle='round', linestyle='-')
            ax.add_patch(rect)

    legend_fs = 15

    ax.set_xlabel('Train data removed (%)')

    if args.ylabel:
        ax.set_ylabel(f'Avg. per-example test loss')

    if args.legend1:
        ax.legend(title='Method', frameon=False, fontsize=legend_fs)

    # add second ghost axis for line styles
    if args.legend2:
        labels = ['Fixed', 'Reestimated']
        styles = ['-', '--']

        ax2 = ax.twinx()
        for style, label in zip(styles, labels):
            ax2.plot(np.NaN, np.NaN, ls=style, label=label, c='black')
        ax2.get_yaxis().set_visible(False)
        ax2.legend(title='Ordering', fontsize=legend_fs, loc='upper left', frameon=False)

    plt_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}')
    suffix = f'_reinf'
    os.makedirs(plt_dir, exist_ok=True)
    fp = os.path.join(plt_dir, f'{args.dataset}')

    plt.tight_layout()
    plt.savefig(fp + suffix + '.pdf', bbox_inches='tight')


def main(args):

    out_dir = os.path.join(args.out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_reinfluence_args().parse_args())
