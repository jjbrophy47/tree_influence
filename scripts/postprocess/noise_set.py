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


def process(args, exp_hash, out_dir, logger):
    color, line, label = util.get_plot_dicts()

    results = []

    for strategy in args.strategy:

        exp_dir = os.path.join(args.in_dir,
                               args.dataset,
                               args.tree_type,
                               f'exp_{exp_hash}',
                               strategy)

        res_list = util.get_results(args, exp_dir, logger, progress_bar=False)
        res_list = util.filter_results(res_list, args.skip)

        for method, res in res_list:
            results.append((f'{label[method]}_{strategy}', res))

    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for method, res in results:

        check_pct = np.array(res['check_frac']) * 100
        ls = '--' if 'self' in method else '-'

        ax = axs[0]
        ax.errorbar(x=check_pct, y=res['frac_detected'] * 100, label=method, linestyle=ls)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('% noisy examples detected')
        ax.set_title(f'Detection')
        ax.legend(fontsize=6)

        ax = axs[1]
        ax.errorbar(x=check_pct, y=res['loss'], label=method, linestyle=ls)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('Test loss')
        ax.set_title(f'Loss')

        ax = axs[2]
        ax.errorbar(x=check_pct, y=res['acc'], label=method, linestyle=ls)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('Test acc.')
        ax.set_title(f'Accuracy')

        ax = axs[3]
        ax.errorbar(x=check_pct, y=res['auc'], label=method, linestyle=ls)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('Test AUC')
        ax.set_title(f'AUC')

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')


def main(args):

    args.method += ['loss']

    exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac,
                'check_frac': args.check_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}')

    log_dir = os.path.join(out_dir, 'logs')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_noise_set_args().parse_args())
