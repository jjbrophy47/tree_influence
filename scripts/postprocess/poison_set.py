"""
Process results for a single dataset.
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


def process(args, exp_dir, out_dir, logger):

    # get dataset
    color, line, label = util.get_plot_dicts()

    res_list = util.get_results(args, exp_dir, logger, progress_bar=False)
    res_list = util.filter_results(res_list, args.skip)

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

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')


def main(args):

    exp_dict = {'poison_frac': args.poison_frac, 'val_frac': args.val_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    exp_dir = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}')

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}')

    log_dir = os.path.join(out_dir, 'postprocess', 'logs')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(f'timestamp: {datetime.now()}')

    process(args, exp_dir, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_poison_set_args().parse_args())
