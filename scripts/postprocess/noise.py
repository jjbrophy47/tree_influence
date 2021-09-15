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

    r1 = {}

    for random_state in range(1, args.n_repeat + 1):
        r1[random_state] = {}

        for strategy in args.strategy:

            exp_dir = os.path.join(args.in_dir,
                                   args.dataset,
                                   args.tree_type,
                                   f'exp_{exp_hash}',
                                   strategy,
                                   f'random_state_{random_state}')

            res_list = util.get_results(args, args.in_dir, exp_dir, logger, progress_bar=False)
            res_list = util.filter_results(res_list, args.skip)

            for method, res in res_list:

                res_simple = {}
                res_simple['frac_detected'] = res['frac_detected']
                res_simple['loss'] = res['loss']
                res_simple['acc'] = res['acc']
                res_simple['auc'] = res['auc']

                name = f'{label[method]}_{strategy}'

                r1[random_state][name] = res_simple

    # average over random states
    fd_list = []
    loss_d_list = []
    acc_d_list = []
    auc_d_list = []
    method_list = []

    temp_dict = {'noise_frac': args.noise_frac, 'check_frac': args.check_frac}
    fd_d = temp_dict.copy()
    loss_d = temp_dict.copy()
    acc_d = temp_dict.copy()
    auc_d = temp_dict.copy()

    for name in r1[random_state].keys():
        fd_d[name] = np.mean(np.vstack([r1[rs][name]['frac_detected'] for rs in r1.keys()]), axis=0)
        loss_d[name] = np.mean(np.vstack([r1[rs][name]['loss'] for rs in r1.keys()]), axis=0)
        acc_d[name] = np.mean(np.vstack([r1[rs][name]['acc'] for rs in r1.keys()]), axis=0)
        auc_d[name] = np.mean(np.vstack([r1[rs][name]['auc'] for rs in r1.keys()]), axis=0)
        method_list.append(name)

    check_frac = np.array(args.check_frac, dtype=np.float32)

    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for method in method_list:

        ax = axs[0]
        ax.errorbar(x=check_frac * 100, y=fd_d[method] * 100, label=method)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('% noisy examples detected')
        ax.set_title(f'Detection')
        ax.legend(fontsize=6)

        ax = axs[1]
        ax.errorbar(x=check_frac * 100, y=loss_d[method], label=method)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('Test loss')
        ax.set_title(f'Loss')

        ax = axs[2]
        ax.errorbar(x=check_frac * 100, y=acc_d[method], label=method)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('Test acc.')
        ax.set_title(f'Accuracy')

        ax = axs[3]
        ax.errorbar(x=check_frac * 100, y=auc_d[method], label=method)
        ax.set_xlabel('% train data checked')
        ax.set_ylabel('Test AUC')
        ax.set_title(f'AUC')

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')


def main(args):

    args.method += ['loss']

    exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
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
    main(post_args.get_noise_args().parse_args())
