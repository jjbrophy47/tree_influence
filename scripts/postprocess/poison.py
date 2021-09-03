"""
Process results and plot them.
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


def process(args, exp_dir, logger):

    # get dataset
    color, line, label = pp_util.get_plot_dicts()

    res_list = filter_results(pp_util.get_results(args, args.in_dir, logger, exp_hash='dummy',
                                                  temp_dir=exp_dir, progress_bar=False), args.skip)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for method, res in res_list:

        ax = axs[0]
        ax.errorbar(res['poison_frac'] * 100, res['loss'], linestyle=line[method],
                    color=color[method], label=label[method])
        ax.axhline(res['loss'][0], color='k', linestyle='--')
        ax.set_ylabel('Test loss')
        ax.set_xlabel('% poison')
        ax.set_title(f'Loss')
        ax.legend(fontsize=6)

        ax = axs[1]
        ax.errorbar(res['poison_frac'] * 100, res['acc'], linestyle=line[method],
                    color=color[method], label=label[method])
        ax.axhline(res['acc'][0], color='k', linestyle='--')
        ax.set_ylabel('Test accuracy')
        ax.set_xlabel('% poison')
        ax.set_title(f'Accuracy')

        ax = axs[2]
        ax.errorbar(res['poison_frac'] * 100, res['auc'], linestyle=line[method],
                    color=color[method], label=label[method])
        ax.axhline(res['auc'][0], color='k', linestyle='--')
        ax.set_ylabel('Test AUC')
        ax.set_xlabel('% poison')
        ax.set_title(f'AUC')

    logger.info(f'\nSaving results to {out_dir}...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')


def main(args):

    exp_dict = {'poison_frac': args.poison_frac, 'val_frac': args.val_frac}
    exp_hash = util.dict_to_hash(exp_dict)

    exp_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}')

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}')

    log_dir = os.path.join(args.out_dir, 'logs')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, exp_dir, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_poison/')
    parser.add_argument('--out_dir', type=str, default='output/plot/poison/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--strategy', type=str, nargs='+', default=['self', 'test_sum'])
    parser.add_argument('--noise', type=str, default='target')
    parser.add_argument('--noise_frac', type=float, default=0.1)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--check_frac', type=float, default=0.1)

    # additional settings
    parser.add_argument('--random_state', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['loss', 'trex', 'similarity', 'similarity', 'boostin2',
                                 'leaf_influenceSP', 'subsample', 'loo', 'target', 'random'])
    parser.add_argument('--skip', type=str, nargs='+', default=[])
    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1, 0])  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default=['dot_prod'])  # Similarity & Similarity2
    parser.add_argument('--measure', type=str, nargs='+', default=['euclidean'])  # InputSimilarity

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
