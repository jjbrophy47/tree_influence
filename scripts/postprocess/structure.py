"""
Evaluate structural changes of LOO.
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

    rng = np.random.default_rng(args.random_state)
    
    assert len(res_list) == 1
    method, res = res_list[0]
    aff = res['affinity']

    # plot
    util.plot_settings(fontsize=14)
    legend_fs = 11.5

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

    # plot 1
    test_idx = 1  # pick a test instance

    ax = axs[0]
    sns.histplot(aff[:, 0, test_idx], ax=ax, label='Initial', element='step', fill=True, color='black')
    sns.histplot(aff[:, 1, test_idx], ax=ax, label='1 removal', element='step', fill=True)
    ax.set_xlabel(r'Affinity $\left(\sum_{t=1}^T 1[R_t(x_i) = R_t(x_e)] \right)$')
    ax.set_ylabel('No. train')
    ax.legend(fontsize=legend_fs)

    # plot 2
    abs_diff = np.abs(aff[:, 0, :] - aff[:, 1, :])  # shape=(no. train, no. test)
    avg_abs_diff = np.mean(abs_diff, axis=1)  # shape=(no. train,)

    ax = axs[1]
    sns.histplot(avg_abs_diff, ax=ax, label='|Initial - 1 removal|', element='step', color='brown')
    ax.set_xlabel('Average difference in affinity')
    ax.set_ylabel('')
    ax.legend(fontsize=legend_fs)

    logger.info(f'\nsaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'structure_{args.tree_type}_{args.dataset}.pdf'), bbox_inches='tight')

    logger.info(f'\ntotal time: {time.time() - begin:.3f}s')


def main(args):

    # get experiment directory
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'n_remove': args.n_remove}
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
    main(post_args.get_structure_args().parse_args())
