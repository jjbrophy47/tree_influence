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

    n_test = None
    reinf_exists = 0

    results = []
    reinf_list = []

    for in_dir in args.in_dir:

        res = pp_util.get_results(args, in_dir, logger)
        res = filter_results(res, args.skip)
        results += res

        is_reinf = 1 if 'reinfluence' in in_dir else 0
        reinf_list += [is_reinf] * len(res)

        if is_reinf and len(res) > 0:
            reinf_exists = 1

    # get dataset
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    color, line, label = pp_util.get_plot_dicts()

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

    exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
                'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = util.dict_to_hash(exp_dict)

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
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, nargs='+', default=['temp_influence/', 'temp_reinfluence/'])
    parser.add_argument('--out_dir', type=str, default='output/plot/roar/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
    parser.add_argument('--n_ckpt', type=int, default=50)

    # additional settings
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['random', 'target', 'similarity', 'boostin2', 'boostin3', 'boostin4',
                                 'trex', 'leaf_influence', 'leaf_influenceSP', 'loo',
                                 'dshap', 'subsample'])  # no minority, loss
    parser.add_argument('--skip', type=str, nargs='+',
                        default=['minority', 'loss'])
    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1])  # LeafInfluence

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
    parser.add_argument('--metric', type=str, nargs='+', default=['mse', 'loss', 'acc', 'auc'])
    parser.add_argument('--std_err', type=int, default=0)
    parser.add_argument('--custom_dir', type=str, default='')
    parser.add_argument('--zoom', type=float, default=0.2)
    parser.add_argument('--plt', type=str, default='no_fill')

    args = parser.parse_args()
    main(args)
