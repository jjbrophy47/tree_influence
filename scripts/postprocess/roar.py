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
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util
from leaf_analysis import filter_results
from config import post_args


def process(args, out_dir, logger):

    n_test = None
    reinf_exists = 0

    results = []
    reinf_list = []

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    for in_dir in args.in_dir:

        exp_dir = os.path.join(in_dir,
                               args.dataset,
                               args.tree_type,
                               f'exp_{exp_hash}')

        res = util.get_results(args, in_dir, exp_dir, logger)
        res = filter_results(res, args.skip)
        results += res

        is_reinf = 1 if 'reinfluence' in in_dir else 0
        reinf_list += [is_reinf] * len(res)

        if is_reinf and len(res) > 0:
            reinf_exists = 1

    # get dataset
    X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, args.dataset)

    color, line, label = util.get_plot_dicts()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    for i, (method, res) in enumerate(results):

        if i == 0:
            n_test = res['loss'].shape[0]

        else:
            temp = res['loss'].shape[0]
            assert n_test == temp, f'Inconsistent no. test: {temp:,} != {n_test:,}'

        is_reinf = reinf_list[i]

        ax = axs[0]

        # TEMP
        if res['remove_frac'].ndim == 1:
            x = res['remove_frac'] * 100
        else:
            x = res['remove_frac'][0] * 100

        y = res['loss'].mean(axis=0)
        y_err = sem(res['loss'], axis=0)
        y_err = y_err if args.std_err else None

        if is_reinf:
            linestyle = '-.'
            labelstyle = f'{label[method]} (RI)'
        else:
            linestyle = line[method]
            labelstyle = label[method]

        if args.plt == 'fill':
            assert y_err is not None

            ax.errorbar(x, y, label=labelstyle, color=color[method], linestyle=linestyle)
            ax.fill_between(x, y - y_err, y + y_err, color=color[method],
                            linestyle=linestyle, alpha=0.2)

        else:
            ax.errorbar(x, y, yerr=y_err, label=labelstyle, color=color[method],
                        linestyle=linestyle, alpha=0.75)

        ax.set_xlabel('Train data removed (%)')
        ax.set_ylabel(f'Avg. example test loss')
        ax.legend(fontsize=6)

        if args.zoom > 0.0 and args.zoom < 1.0:
            ax = axs[1]

            n = int(len(x) * args.zoom)
            x, y = x[:n], y[:n]

            if y_err is not None:
                y_err = y_err[:n]

            if args.plt == 'fill':
                assert y_err is not None

                ax.errorbar(x, y, label=labelstyle, color=color[method], linestyle=linestyle)
                ax.fill_between(x, y - y_err, y + y_err, label=labelstyle, color=color[method],
                                linestyle=linestyle, alpha=0.2)
            else:
                ax.errorbar(x, y, yerr=y_err, label=labelstyle, color=color[method],
                            linestyle=linestyle, alpha=0.75)

            ax.set_xlabel('Train data removed (%)')
            ax.set_ylabel(f'Avg. example test loss')

    if args.zoom <= 0.0 or args.zoom >= 1.0:
        fig.delaxes(axs[1])

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    custom_dir = 'reestimation' if reinf_exists else args.custom_dir

    plt_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           custom_dir)
    suffix = f'_{n_test}'
    os.makedirs(plt_dir, exist_ok=True)
    fp = os.path.join(plt_dir, f'{args.dataset}')

    plt.tight_layout()
    plt.savefig(fp + suffix + '.png', bbox_inches='tight')
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
    main(post_args.get_roar_args().parse_args())
