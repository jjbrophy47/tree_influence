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
from leaf_analysis import filter_results


def process(args, out_dir, logger):

    results = []

    for strategy in args.strategy:
        exp_dict = {'inf_obj': args.inf_obj, 'strategy': strategy,
                    'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
        exp_hash = util.dict_to_hash(exp_dict)

        res = pp_util.get_results(args, args.in_dir, logger, exp_hash=exp_hash)
        res = filter_results(res, args.skip)

        results += [tup + (strategy,) for tup in res]

    # get dataset
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    color, line, label = pp_util.get_plot_dicts()

    n_row = 2 if args.zoom > 0.0 and args.zoom < 1.0 else 1
    height = 8 if args.zoom > 0.0 and args.zoom < 1.0 else 4

    fig, axs = plt.subplots(n_row, 3, figsize=(12, height))
    axs = axs.flatten()

    for i, metric in enumerate(args.metric):

        for method, res, strategy in results:
            ax = axs[i]

            linestyle = '-'
            if 'random' in method or 'minority' in method:
                linestyle = ':'
            elif strategy == 'self':
                linestyle = '-.'
            method_color = 'gray' if 'boostin4' in method else color[method]

            x, y = res['remove_frac'] * 100, res[metric]

            ax.plot(x, y, label=label[method], color=method_color,
                    linestyle=linestyle, alpha=0.75)
            ax.set_xlabel('Train data removed (%)')
            ax.set_ylabel(f'Test {metric}')

            # plot zoomed version
            if args.zoom > 0.0 and args.zoom < 1.0:
                ax = axs[i + 3]

                n = int(len(x) * args.zoom)
                x, y = x[:n], y[:n]

                ax.plot(x, y, label=label[method], color=method_color,
                        linestyle=linestyle, alpha=0.75)
                ax.set_xlabel('Train data removed (%)')
                ax.set_ylabel(f'Test {metric}')

        if y[0] == -1:
            fig.delaxes(ax=axs[i])

            if args.zoom > 0.0 and args.zoom < 1.0:
                fig.delaxes(ax=axs[i + 3])

        elif i == 0 or i == 3:
            axs[i].legend(fontsize=6)

    ax2 = axs[0].twinx()
    ax2.plot(np.NaN, np.NaN, ls='-', label='test_sum', c='k')
    ax2.plot(np.NaN, np.NaN, ls='-.', label='self-inf.', c='k')
    ax2.plot(np.NaN, np.NaN, ls=':', label='neither', c='k')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(fontsize=6)

    plt_dir = os.path.join(args.out_dir,
                           args.inf_obj,
                           args.tree_type)
    suffix = ''
    os.makedirs(plt_dir, exist_ok=True)
    fp = os.path.join(plt_dir, f'{args.dataset}')

    plt.tight_layout()
    plt.savefig(fp + suffix + '.png', bbox_inches='tight')
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
    parser.add_argument('--in_dir', type=str, default='temp_influence2/')
    parser.add_argument('--out_dir', type=str, default='output/plot/roar2/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--strategy', type=str, nargs='+', default=['test_sum', 'self'])
    parser.add_argument('--remove_frac', type=float, default=0.5)
    parser.add_argument('--n_ckpt', type=int, default=50)

    # additional settings
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['random', 'target', 'similarity', 'boostin2', 'boostin4',
                                 'trex', 'leaf_influence', 'leaf_influenceSP', 'loo',
                                 'subsample', 'minority'])
    parser.add_argument('--skip', type=str, nargs='+',
                        default=['loss'])
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

    # result settings
    parser.add_argument('--metric', type=str, nargs='+', default=['loss', 'acc', 'auc'])
    parser.add_argument('--std_err', type=int, default=0)
    parser.add_argument('--custom_dir', type=str, default='')
    parser.add_argument('--zoom', type=float, default=-1)
    parser.add_argument('--plt', type=str, default='no_fill')

    args = parser.parse_args()
    main(args)
