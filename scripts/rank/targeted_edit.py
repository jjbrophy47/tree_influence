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
import matplotlib.ticker as ticker
from scipy.stats import sem
from scipy.stats import gmean
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import rank_args
from remove import get_mean_df
from remove import plot_mean_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    df_list = []
    df_li_list = []
    df_rel_list = []
    df_edit_list = []

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

            if ckpt == 1:
                fp_edit = os.path.join(ckpt_dir, 'edit_frac.csv')
                assert os.path.exists(fp_edit), f'{fp_edit} does not exist!'
                df_edit_list.append(pd.read_csv(fp_edit))

    df_all = pd.concat(df_list)
    df_li_all = pd.concat(df_li_list)
    df_rel_all = pd.concat(df_rel_list)
    df_edit_all = pd.concat(df_edit_list) * 100

    # average ranks among different checkpoints and/or tree types
    group_cols = ['dataset']

    df_all = df_all.groupby(group_cols).mean().reset_index()
    df_li_all = df_li_all.groupby(group_cols).mean().reset_index()
    df_rel_all = df_rel_all.groupby(group_cols).mean().reset_index()
    df_edit_all = df_edit_all.groupby(group_cols).mean().reset_index()

    # compute average ranks
    skip_cols = ['dataset', 'tree_type', 'edit_frac']
    li_cols = ['LeafInfluence', 'LeafInfluenceLE', 'LeafRefit', 'LeafRefitLE']

    df = get_mean_df(df_all, skip_cols=skip_cols, sort='ascending')
    df_li = get_mean_df(df_li_all, skip_cols=skip_cols, sort='ascending')
    df_rel = get_mean_df(df_rel_all, skip_cols=skip_cols + li_cols, sort='descending', geo_mean=True)
    df_rel_li = get_mean_df(df_rel_all, skip_cols=skip_cols, sort='descending', geo_mean=True)
    df_edit = get_mean_df(df_edit_all, skip_cols=skip_cols + li_cols, sort='ascending')
    df_edit_li = get_mean_df(df_edit_all, skip_cols=skip_cols, sort='ascending')

    logger.info(f'\nLoss (ranking):\n{df}')
    logger.info(f'\nLoss (ranking-LI):\n{df_li}')
    logger.info(f'\nLoss (relative):\n{df_rel}')
    logger.info(f'\nLoss (relative-LI):\n{df_rel_li}')
    logger.info(f'\nEdit fraction:\n{df_edit}')
    logger.info(f'\nEdit fraction (LI):\n{df_edit_li}')

    # plot
    n_datasets = len(df_all['dataset'].unique())
    n_li_datasets = len(df_li_all['dataset'].unique())

    label_dict = {'Target': 'RandomSL'}

    df = df.rename(index=label_dict)
    df_li = df_li.rename(index=label_dict)

    df_rel = df_rel.rename(index=label_dict)
    df_rel_li = df_rel_li.rename(index=label_dict)

    # reorder methods
    order = ['BoostInLE', 'LeafInfSPLE', 'TREX', 'TreeSim', 'SubSample', 'LOOLE']
    order_li = ['BoostInLE', 'LeafInfSPLE', 'TREX', 'TreeSim', 'LeafRefitLE', 'LeafInfluenceLE',
                'SubSample', 'LOOLE']

    if 'RandomSL' in df.index:
        order.append('RandomSL')
        order_li.append('RandomSL')

    df = df.reindex(order)
    df_li = df_li.reindex(order_li)
    df_rel = df_rel.reindex(order)
    df_rel_li = df_rel_li.reindex(order_li)
    df_edit = df_edit.reindex(order)
    df_edit_li = df_edit_li.reindex(order_li)

    df.index = df.index.str.replace('LE', '')
    df_li.index = df_li.index.str.replace('LE', '')
    df_rel.index = df_rel.index.str.replace('LE', '')
    df_rel_li.index = df_rel_li.index.str.replace('LE', '')
    df_edit.index = df_edit.index.str.replace('LE', '')
    df_edit_li.index = df_edit_li.index.str.replace('LE', '')

    labels = [x for x in df.index]
    labels_li = [x for x in df_li.index]

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

    df_edit.to_csv(os.path.join(out_dir, 'edit_frac.csv'))
    df_edit_li.to_csv(os.path.join(out_dir, 'edit_frac_li.csv'))

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
    main(rank_args.get_targeted_edit_args().parse_args())
