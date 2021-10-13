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


def get_mean_rank_df(in_df, skip_cols=[], sort=None):
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

    # 95% CI
    df = pd.DataFrame(np.vstack([in_df.mean(axis=0), 1.96 * in_df.sem(axis=0)]).T,
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
    df_time_list = []
    df_mem_list = []

    for tree_type in args.tree_type:

        in_dir = os.path.join(args.in_dir,
                              tree_type,
                              f'exp_{exp_hash}',
                              'summary')

        # get resource usage
        ckpt_dir = os.path.join(in_dir, f'ckpt_{args.ckpt[0]}')

        fp_time = os.path.join(ckpt_dir, 'runtime.csv')
        fp_mem = os.path.join(ckpt_dir, 'mem.csv')
        assert os.path.exists(fp_time), f'{fp_time} does not exist!'
        assert os.path.exists(fp_mem), f'{fp_mem} does not exist!'

        df_time_list.append(pd.read_csv(fp_time))
        df_mem_list.append(pd.read_csv(fp_mem))

        # get loss
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
    df_time_all = pd.concat(df_time_list)
    df_mem_all = pd.concat(df_mem_list)

    # average ranks among different checkpoints and/or tree types
    group_cols = ['dataset']

    df_all = df_all.groupby(group_cols).mean().reset_index()
    df_li_all = df_li_all.groupby(group_cols).mean().reset_index()
    df_time_all = df_time_all.groupby(group_cols).mean().reset_index()
    df_mem_all = df_mem_all.groupby(group_cols).mean().reset_index()

    # compute average ranks
    skip_cols = ['dataset', 'tree_type', 'remove_frac']

    df = get_mean_rank_df(df_all, skip_cols=skip_cols, sort='ascending')
    df_li = get_mean_rank_df(df_li_all, skip_cols=skip_cols, sort='ascending')

    logger.info(f'\nLoss:\n{df}')
    logger.info(f'\nLoss (li):\n{df_li}')

    # # combine dataframes
    # index = df_li.index
    # df = df_li.reset_index().merge(df.reset_index(), on='index', how='left')
    # means_df = df[['index', 'mean_x', 'mean_y']].copy()
    # sems_df = df[['index', 'sem_x', 'sem_y']].copy()

    # # rename and clean up
    # means_df.index = means_df['index']
    # sems_df.index = means_df['index']
    # del means_df['index']
    # del sems_df['index']
    # means_df.columns = ['Subgroup A', 'All datasets']
    # sems_df.columns = ['Subgroup A', 'All datasets']

    # print(means_df)
    # print(sems_df)

    # # plot
    # fig, ax = plt.subplots(figsize=(4, 4))
    # means_df.plot.bar(yerr=sems_df, ax=ax, rot=45,
    #                   title=f'Loss ({len(means_df)} datasets)', capsize=3,
    #                   ylabel='Avg. rank', xlabel='Method')

    # plot
    n_datasets = len(df_all['dataset'].unique())
    n_li_datasets = len(df_li_all['dataset'].unique())

    df = df.rename(columns={'mean': 'All datasets'}, index={'LeafInfluence': 'LeafInf.', 'SubSample': 'SubS.'})
    df_li = df_li.rename(columns={'mean': 'Subgroup A'}, index={'LeafInfluence': 'LeafInf.', 'SubSample': 'SubS.'})

    labels = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(df.index)]
    labels_li = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(df_li.index)]

    pp_util.plot_settings(fontsize=28)
    width = 22
    height = pp_util.get_height(width, subplots=(1, 2))

    fig, axs = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': [6, 8]})

    ax = axs[0]
    df.plot(kind='bar', y='All datasets', yerr='sem', ax=ax, title=None, capsize=3,
            ylabel='Average rank', xlabel=None, legend=True, color='blueviolet')
    ax.set_xticklabels(labels, rotation=0)

    ax = axs[1]
    df_li.plot(kind='bar', y='Subgroup A', yerr='sem', ax=ax, title=None, capsize=3,
               ylabel=None, xlabel=None, legend=True, color='mediumseagreen')
    ax.set_xticklabels(labels_li, rotation=0)

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roar.pdf'), bbox_inches='tight')
    plt.show()

    df.to_csv(os.path.join(out_dir, 'loss_rank.csv'))
    df_li.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'))

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    assert len(args.tree_type) > 0
    out_dir = os.path.join(args.in_dir,
                           'rank',
                           f'exp_{exp_hash}',
                           f'+'.join(args.tree_type))

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(rank_args.get_roar_args().parse_args())
