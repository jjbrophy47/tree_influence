"""
Rank summarization results.
"""
import os
import sys
import time
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import gmean
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import rank_args


def get_mean_rank(in_df, skip_cols=[], sort=None):
    """
    Compute mean rank (with sem) for each method.

    Input
        df: pd.DataFrame, dataframe with values to compute mean over.
        skip_cols: list, list of columns to ignore.
        sort: str, Sort methods; 'ascending', 'descending', or None.

    Return new pd.DataFrame with the original columns as the index.
    """
    cols = [c for c in list(in_df.columns) if c not in skip_cols]

    in_df = in_df[cols]

    df = pd.DataFrame(np.vstack([in_df.mean(axis=0), in_df.sem(axis=0)]).T,
                      index=cols, columns=['mean', 'sem'])

    if sort == 'ascending':
        df = df.sort_values('mean')

    elif sort == 'descending':
        df = df.sort_values('mean', ascending=False)

    return df


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

        for ckpt in args.ckpt:
            ckpt_dir = os.path.join(in_dir, f'ckpt_{ckpt}')

            fp = os.path.join(ckpt_dir, 'loss_rank.csv')
            fp_li = os.path.join(ckpt_dir, 'loss_rank_li.csv')
            assert os.path.exists(fp), f'{fp} does not exist!'
            assert os.path.exists(fp_li), f'{fp_li} does not exist!'

            df_list.append(pd.read_csv(fp))
            df_li_list.append(pd.read_csv(fp_li))

    df_all = pd.concat(df_list)
    df_li_all = pd.concat(df_li_list)

    n_datasets = len(df_all['dataset'].unique())
    n_li_datasets = len(df_li_all['dataset'].unique())

    df = get_mean_rank(df_all, skip_cols=['dataset', 'remove_frac'], sort='ascending')
    df_li = get_mean_rank(df_li_all, skip_cols=['dataset', 'remove_frac'], sort='ascending')

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    ax = axs[0]
    df.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45,
            title=f'{n_datasets} datasets',
            ylabel='Avg. rank', xlabel='Method', legend=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1]
    df_li.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45,
               title=f'{n_li_datasets} datasets',
               ylabel='Avg. rank', xlabel='Method', legend=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.savefig(os.path.join(out_dir, 'loss_rank.png'), bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    df.to_csv(os.path.join(out_dir, 'loss_rank.csv'))
    df_li.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'))

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
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

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(rank_args.get_roar_args().parse_args())
