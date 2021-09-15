"""
Evaluate min. no. train examples to edit to flip test prediction.
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
    res_list = util.get_results(args, args.in_dir, exp_dir, logger=logger)
    res_list = util.filter_results(res_list, args.skip)

    color, line, label = util.get_plot_dicts()

    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    
    for method, res in res_list:

        df = res['df']
        n_correct = len(df[df['status'] == 'success']) + len(df[df['status'] == 'fail'])

        df = df[df['status'] == 'success']
        if len(df) == 0:
            continue

        df = df.sort_values('n_edits')
        df['count'] = np.arange(1, len(df) + 1)

        ax.plot(df['frac_edits'] * 100, df['count'], label=label[method], color=color[method],
                linestyle=line[method], alpha=0.75)
        ax.set_xlabel('% train targets poisoned')
        ax.set_ylabel(f'No. correct test preds. flipped (cum.)')

        ax2.plot(df['n_edits'], df['count'], label=label[method], color=color[method],
                 linestyle=line[method], alpha=0.75)
        ax2.set_xlabel('No. train targets poisoned')

    ax.axhline(n_correct, label='No. correct test preds.', color='k', linestyle='--', linewidth=1)
    ax.legend(fontsize=6)

    logger.info(f'\nsaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')

    logger.info(f'\ntotal time: {time.time() - begin:.3f}s')


def main(args):

    # get experiment directory
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac,
                'n_ckpt': args.n_ckpt, 'step_size': args.step_size}
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
    main(post_args.get_counterfactual_args().parse_args())
