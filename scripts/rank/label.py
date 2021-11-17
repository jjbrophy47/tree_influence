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
from remove import get_mean_df


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

    # average ranks among different checkpoints and/or tree types
    group_cols = ['dataset']

    df_all = df_all.groupby(group_cols).mean().reset_index()
    df_li_all = df_li_all.groupby(group_cols).mean().reset_index()
    df_rel_all = df_rel_all.groupby(group_cols).mean().reset_index()

    # compute average ranks
    skip_cols = ['dataset', 'tree_type', 'edit_frac']

    df = get_mean_df(df_all, skip_cols=skip_cols, sort='ascending')
    df_li = get_mean_df(df_li_all, skip_cols=skip_cols, sort='ascending')
    df_rel = get_mean_df(df_rel_all, skip_cols=skip_cols + ['LeafInfluence', 'LeafRefit'],
                         sort='descending', geo_mean=True)
    df_rel_li = get_mean_df(df_rel_all, skip_cols=skip_cols,
                            sort='descending', geo_mean=True)

    logger.info(f'\nLoss (ranking):\n{df}')
    logger.info(f'\nLoss (ranking-LI):\n{df_li}')
    logger.info(f'\nLoss (relative):\n{df_rel}')
    logger.info(f'\nLoss (relative-LI):\n{df_rel_li}')

    # plot
    n_datasets = len(df_all['dataset'].unique())
    n_li_datasets = len(df_li_all['dataset'].unique())

    label_dict = {'LeafInfluence': 'LeafInf.', 'SubSample': 'SubS.', 'Target': 'RandomSL'}

    df = df.rename(columns={'mean': 'All datasets'}, index=label_dict)
    df_li = df_li.rename(columns={'mean': 'SDS'}, index=label_dict)

    df_rel = df_rel.rename(index=label_dict)
    df_rel_li = df_rel_li.rename(index=label_dict)

    # reorder methods
    order = ['BoostIn', 'BoostInW1', 'BoostInW2', 'LeafInfSP', 'TreeSim',
             'TREX', 'SubS.', 'LOO', 'RandomSL', 'Random']
    order_li = ['LeafRefit', 'LeafInf.', 'BoostIn', 'BoostInW1', 'BoostInW2',
                'LeafInfSP', 'TreeSim', 'TREX', 'SubS.', 'LOO', 'RandomSL', 'Random']

    df = df.reindex(order)
    df_li = df_li.reindex(order_li)

    labels = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(df.index)]
    labels_li = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(df_li.index)]

    pp_util.plot_settings(fontsize=28)
    width = 22
    height = pp_util.get_height(width, subplots=(1, 2))

    fig, axs = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': [6, 8]})

    ax = axs[0]
    df.plot(kind='bar', y='All datasets', yerr='sem', ax=ax, title=None, capsize=3,
            ylabel='Average rank', xlabel=None, legend=True, color='#3e9ccf')
    ax.set_xticklabels(labels, rotation=0)

    ax = axs[1]
    df_li.plot(kind='bar', y='SDS', yerr='sem', ax=ax, title=None, capsize=3,
               ylabel=None, xlabel=None, legend=True, color='#ff7600')
    ax.set_xticklabels(labels_li, rotation=0)

    ax.axvline(1.5, color='gray', linestyle='--')
    # ax.axvline(7.5, color='gray', linestyle='--')

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'label.pdf'), bbox_inches='tight')

    df.to_csv(os.path.join(out_dir, 'loss_rank.csv'))
    df_li.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'))

    df_rel.to_csv(os.path.join(out_dir, 'loss_rel.csv'))
    df_rel_li.to_csv(os.path.join(out_dir, 'loss_rel_li.csv'))

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'n_test': args.n_test, 'edit_frac': args.edit_frac}
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
    main(rank_args.get_label_args().parse_args())
