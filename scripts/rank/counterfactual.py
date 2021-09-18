"""
Rank summarization results.
"""
import os
import sys
import time
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.metrics import log_loss

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import rank_args
from rank.roar import get_mean_rank_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    df_list = []
    df_li_list = []

    for tree_type in args.tree_type:

            in_dir = os.path.join(args.in_dir,
                                  tree_type,
                                  f'exp_{exp_hash}',
                                  'summary')

            fp = os.path.join(in_dir, 'frac_edits_rank.csv')
            fp_li = os.path.join(in_dir, 'frac_edits_rank_li.csv')
            assert os.path.exists(fp), f'{fp} does not exist!'
            assert os.path.exists(fp_li), f'{fp_li} does not exist!'

            df_list.append(pd.read_csv(fp))
            df_li_list.append(pd.read_csv(fp_li))

    df_all = pd.concat(df_list)
    df_li_all = pd.concat(df_li_list)

    # compute average rankings
    skip_cols = ['dataset', 'tree_type']

    df = get_mean_rank_df(df_all, skip_cols=skip_cols, sort='ascending')
    df_li = get_mean_rank_df(df_li_all, skip_cols=skip_cols, sort='ascending')

    # plot
    n_datasets = len(df_all['dataset'].unique())
    n_li_datasets = len(df_li_all['dataset'].unique())

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    ax = axs[0]
    df.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45,
            title=f'Frac. edits ({n_datasets} datasets)', capsize=3,
            ylabel='Avg. rank', xlabel='Method', legend=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1]
    df_li.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45,
               title=f'w/ LeafInf ({n_li_datasets} datasets)', capsize=3,
               ylabel='Avg. rank', xlabel='Method', legend=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.savefig(os.path.join(out_dir, 'rank.png'), bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    df.to_csv(os.path.join(out_dir, 'frac_edits_rank.csv'))
    df_li.to_csv(os.path.join(out_dir, 'frac_edits_rank_li.csv'))

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac,
                'n_ckpt': args.n_ckpt, 'step_size': args.step_size}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    if len(args.tree_type) == 1:
        out_dir = os.path.join(args.in_dir,
                               args.tree_type[0],
                               f'exp_{exp_hash}',
                               'summary',
                               'rank')

    else:
        assert len(args.tree_type) > 1
        out_dir = os.path.join(args.in_dir,
                               'rank',
                               f'exp_{exp_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(rank_args.get_counterfactual_args().parse_args())
