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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import sem
from scipy.stats import gmean
from scipy.stats import gstd
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import rank_args


def get_mean_df(in_df, skip_cols=[], sort=None, geo_mean=False):
    """
    Compute mean (with sem) for each method.

    Input
        df: pd.DataFrame, dataframe with values to compute mean over.
        skip_cols: list, list of columns to ignore.
        sort: str, Sort methods; 'ascending', 'descending', or None.

    Return new pd.DataFrame with the original columns as the index.
    """
    cols = [c for c in list(in_df.columns) if c not in skip_cols]

    in_df = in_df[cols].dropna()

    if geo_mean:  # geometric mean and geometric std. dev.
        vals = np.log(in_df.values)
        means = np.exp(np.mean(vals, axis=0))
        sems = np.exp(1.96 * sem(vals, axis=0))  # 95% CI
        # means = gmean(in_df.values, axis=0)
        # sems = gstd(in_df.values, axis=0)

    else:  # arithmetic mean and 95% CI
        means = np.mean(in_df.values, axis=0)
        sems = 1.96 * in_df.sem(axis=0)  # 95% CI

    df = pd.DataFrame(np.vstack([means, sems]).T, index=cols, columns=['mean', 'sem'])

    if sort == 'ascending':
        df = df.sort_values('mean')

    elif sort == 'descending':
        df = df.sort_values('mean', ascending=False)

    return df


def plot_mean_df(df, df_li, out_dir, fn='rank', ylabel='', yerr='sem', add_width=0, add_height=0):
    """
    Plot mean rank or magnitude results.

    Input
        df: All datasets results.
        df_li: SDS results.
    """
    pp_util.plot_settings(fontsize=23)

    labels = [x for x in df.index]
    labels_li = [x for x in df_li.index]

    cmap = matplotlib.cm.get_cmap('Blues')
    cmap_li = matplotlib.cm.get_cmap('Oranges')

    ticks_y = ticker.FuncFormatter(lambda x, pos: f'{x:.1f}x')
    xlabel = 'Influence method'

    fig, axs = plt.subplots(2, 1, figsize=(6 + add_width, 9 + add_height))

    ax = axs[0]
    df_li.plot(kind='bar', y='mean', yerr=yerr, ax=ax, title=None, capsize=3,
               ylabel=f'{ylabel}', xlabel=None, legend=False, color=cmap_li(0.69))
    ax.set_xticklabels(labels_li, rotation=45, ha='right')
    ax.set_ylim(1.0, None)

    ax = axs[1]
    df.plot(kind='bar', y='mean', yerr=yerr, ax=ax, title=None, capsize=3,
            ylabel=ylabel, xlabel=xlabel, legend=False, color=cmap(0.69))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(1.0, None)

    if 'magnitude' in fn:
        axs[0].yaxis.set_major_formatter(ticks_y)
        axs[1].yaxis.set_major_formatter(ticks_y)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{fn}.pdf'), bbox_inches='tight')


def process(args, exp_hash, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    df_list = []
    df_li_list = []
    df_rel_list = []

    for tree_type in args.tree_type:

        in_dir = os.path.join(args.in_dir,
                              tree_type,
                              f'exp_{exp_hash}',
                              'summary')

        # get resource usage
        ckpt_dir = os.path.join(in_dir, f'ckpt_{args.ckpt[0]}')

        # get loss
        for ckpt in args.ckpt:
            ckpt_dir = os.path.join(in_dir, f'ckpt_{ckpt}')

            # ranking
            fp = os.path.join(ckpt_dir, 'loss_rank.csv')
            fp_li = os.path.join(ckpt_dir, 'loss_rank_li.csv')
            assert os.path.exists(fp), f'{fp} does not exist!'
            assert os.path.exists(fp_li), f'{fp_li} does not exist!'

            df_list.append(pd.read_csv(fp))
            df_li_list.append(pd.read_csv(fp_li))

            # relative increase
            fp_rel = os.path.join(ckpt_dir, 'loss_rel.csv')
            assert os.path.exists(fp_rel), f'{fp_rel} does not exist!'

            df_rel_list.append(pd.read_csv(fp_rel))

    df_all = pd.concat(df_list)
    df_li_all = pd.concat(df_li_list)
    df_rel_all = pd.concat(df_rel_list)

    # average among different checkpoints and/or tree types
    group_cols = ['dataset']

    df_all = df_all.groupby(group_cols).mean().reset_index()
    df_li_all = df_li_all.groupby(group_cols).mean().reset_index()
    df_rel_all = df_rel_all.groupby(group_cols).mean().reset_index()

    # compute average ranks and relative performances
    skip_cols = ['dataset', 'tree_type', 'remove_frac']
    remove_cols = ['LeafInfluence', 'LeafRefit']

    df = get_mean_df(df_all, skip_cols=skip_cols, sort='ascending')
    df_li = get_mean_df(df_li_all, skip_cols=skip_cols, sort='ascending')
    df_rel = get_mean_df(df_rel_all, skip_cols=skip_cols + remove_cols, sort='descending', geo_mean=True)
    df_rel_li = get_mean_df(df_rel_all, skip_cols=skip_cols, sort='descending', geo_mean=True)

    logger.info(f'\nLoss (ranking):\n{df}')
    logger.info(f'\nLoss (ranking-LI):\n{df_li}')
    logger.info(f'\nLoss (relative):\n{df_rel}')
    logger.info(f'\nLoss (relative-LI):\n{df_rel_li}')

    # plot
    n_datasets = len(df_all['dataset'].unique())
    n_li_datasets = len(df_li_all['dataset'].unique())

    label_dict = {'Target': 'RandomSL'}

    df = df.rename(index=label_dict)
    df_li = df_li.rename(index=label_dict)

    df_rel = df_rel.rename(index=label_dict)
    df_rel_li = df_rel_li.rename(index=label_dict)

    # reorder methods
    order = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim', 'SubSample', 'LOO']
    order_li = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim', 'LeafInfluence', 'LeafRefit', 'SubSample', 'LOO']

    if 'RandomSL' in df.index:
        order.append('RandomSL')
        order_li.append('RandomSL')

    df = df.reindex(order)
    df_li = df_li.reindex(order_li)
    df_rel = df_rel.reindex(order)
    df_rel_li = df_rel_li.reindex(order_li)

    logger.info(f'\nSaving results to {out_dir}/...')

    height = 2
    plot_mean_df(df, df_li, out_dir=out_dir, fn='loss_rank', ylabel='Avg. rank', add_height=height)
    plot_mean_df(df_rel, df_rel_li, out_dir=out_dir, fn='loss_magnitude', yerr=None,
                 ylabel=r'Gmean. loss $\uparrow$' '\n(rel. to Random)', add_height=height)

    # CSVs
    df.to_csv(os.path.join(out_dir, 'loss_rank.csv'))
    df_li.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'))

    df_rel.to_csv(os.path.join(out_dir, 'loss_rel.csv'))
    df_rel_li.to_csv(os.path.join(out_dir, 'loss_rel_li.csv'))

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
    main(rank_args.get_remove_args().parse_args())
