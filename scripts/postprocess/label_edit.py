"""
Label Edit.
"""
import os
import sys
import time
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.metrics import log_loss

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from experiments import util as exp_util
from config import post_args


def experiment(args, exp_dir, out_dir, logger):
    begin = time.time()

    # get results
    res_list = util.get_results(args, exp_dir, logger=logger)
    res_list = util.filter_results(res_list, args.skip)

    color, line, label = util.get_plot_dicts()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    for method, res in res_list:

        # extract results
        edit_frac = res['edit_frac']  # shape=(no. ckpts.)
        loss = res['loss']  # shape=(no. test, no. ckpts.)
        affinity = res['affinity']  # shape=(2, no. train, no. test)

        # take per-example avg.
        avg_loss = np.mean(loss, axis=0)  # shape=(no. ckpts.)

        ax = axs[0]
        ax.plot(edit_frac * 100, avg_loss, label=label[method], color=color[method],
                linestyle=line[method], alpha=0.75)
        ax.set_xlabel('% train targets edited')
        ax.set_ylabel('Per-example test loss')
        ax.legend(fontsize=6)

        # plot initial affinity
        ax = axs[1]

        avg_affinity = np.mean(affinity, axis=2)

        if 'boostin' in method or 'leaf_sim' in method:
            sns.histplot(avg_affinity[0], ax=ax, label=label[method], element='step', fill=True, color=color[method])
            ax.set_xlabel('Avg. Affinity')
            ax.set_ylabel('No. train')
            ax.set_title(f'Affinity - Initial')
            ax.legend(fontsize=6)

        # plot affinity after 1% label edits
        ax = axs[2]

        avg_affinity = np.mean(affinity, axis=2)

        if 'boostin' in method or 'leaf_sim' in method:
            sns.histplot(avg_affinity[1], ax=ax, label=label[method], element='step', fill=True, color=color[method])
            ax.set_xlabel('Avg. Affinity')
            ax.set_ylabel('No. train')
            ax.set_title(f'Affinity - After {res["affinity_edit_frac"][1] * 100:.0f}% Edits')

    logger.info(f'\nsaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')

    logger.info(f'\ntotal time: {time.time() - begin:.3f}s')


def main(args):

    # get experiment directory
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'step_size': args.step_size}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    exp_dir = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}')

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}')
    log_dir = os.path.join(out_dir, 'logs')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, exp_dir, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_label_edit_args().parse_args())
