"""
Targeted label postprocessing.
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
    args.method = args.method_list
    res_list = util.get_results(args, exp_dir, logger=logger)
    res_list = util.filter_results(res_list, args.skip)

    color, line, label = util.get_plot_dicts()

    fig, ax = plt.subplots()
    
    for method, res in res_list:

        # extract results
        edit_frac = res['edit_frac']  # shape=(no. ckpts.)
        loss = res['loss']  # shape=(no. test, no. ckpts.)

        avg_loss = np.mean(loss, axis=0)  # shape=(no. ckpts.,)s

        # take per-example avg.
        ax.plot(edit_frac * 100, avg_loss, label=label[method], color=color[method],
                linestyle=line[method], alpha=0.75)
        ax.set_xlabel('% train targets edited')
        ax.set_ylabel('Per-example test loss')
        ax.set_title('Loss')
        ax.legend(fontsize=6)

    logger.info(f'\nsaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')

    logger.info(f'\ntotal time: {time.time() - begin:.3f}s')


def main(args):

    # get experiment directory
    exp_dict = {'n_test': args.n_test, 'edit_frac': args.edit_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    in_dir = os.path.join(args.in_dir,
                          args.dataset,
                          args.tree_type,
                          f'exp_{exp_hash}')

    print(in_dir)

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

    experiment(args, in_dir, out_dir, logger)


if __name__ == '__main__':
    main(post_args.get_targeted_edit_args().parse_args())
