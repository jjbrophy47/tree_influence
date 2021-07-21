"""
Aggregate results and organize them into one dict.
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
import util as pp_util
from experiments import util


def process(args, out_dir, logger):

    results = pp_util.get_results(args, logger)
    color, line, label = pp_util.get_plot_dicts()

    if args.inf_obj == 'global':
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        for i, metric in enumerate(args.metric):
            ax = axs[i]
            if i == 1:
                ax.set_title(f'{args.dataset}')

            for method, res in results:
                x, y = res['remove_frac'] * 100, res[metric]

                if args.zoom > 0.0 and args.zoom < 1.0:
                    n = int(len(x) * args.zoom)
                    x, y = x[:n], y[:n]

                ax.plot(x, y, label=label[method], color=color[method],
                        linestyle=line[method], alpha=0.75)
                ax.set_xlabel('Train data removed (%)')
                ax.set_ylabel(f'Test {metric}')
                ax.legend(fontsize=6)

            if y[0] == -1:
                fig.delaxes(ax)

    # local
    else:
        fig, ax = plt.subplots()
        for method, res in results:
            x, y = res['remove_frac'].mean(axis=0) * 100, res['loss'].mean(axis=0)
            y_err = sem(res['loss'], axis=0)
            y_err = y_err if args.std_err else None

            if args.zoom > 0.0 and args.zoom < 1.0:
                n = int(len(x) * args.zoom)
                x, y, y_err = x[:n], y[:n], y_err[:n]

            ax.errorbar(x, y, yerr=y_err, label=label[method], color=color[method],
                        linestyle=line[method], alpha=0.75)
            ax.set_title(f'{args.dataset}')
            ax.set_xlabel('Train data removed (%)')
            ax.set_ylabel(f'Avg. example test loss')
            ax.legend(fontsize=6)

    plt_dir = os.path.join(args.out_dir, args.inf_obj)
    if args.zoom > 0.0 and args.zoom < 1.0:
        plt_dir = os.path.join(plt_dir, 'zoom')
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
    parser.add_argument('--in_dir', type=str, default='output/roar/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plot/roar/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='global')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['random', 'minority', 'boostin', 'trex', 'leaf_influence', 'loo', 'dshap'])
    parser.add_argument('--use_leaf', type=int, nargs='+', default=[1, 0])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1, 0])  # LeafInfluence

    parser.add_argument('--kernel', type=str, nargs='+', default=['lpw'])  # Trex
    parser.add_argument('--target', type=str, nargs='+', default=['actual'])  # Trex
    parser.add_argument('--lmbd', type=float, nargs='+', default=[0.003])  # Trex
    parser.add_argument('--n_epoch', type=str, nargs='+', default=[3000])  # Trex

    parser.add_argument('--trunc_frac', type=float, nargs='+', default=[0.25])  # DShap
    parser.add_argument('--check_every', type=int, nargs='+', default=[100])  # DShap

    parser.add_argument('--global_op', type=str, nargs='+', default=['self', 'expected'])  # TREX, LOO, DShap

    # result settings
    parser.add_argument('--metric', type=str, nargs='+', default=['mse', 'loss', 'acc', 'auc'])
    parser.add_argument('--std_err', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)

    args = parser.parse_args()
    main(args)
