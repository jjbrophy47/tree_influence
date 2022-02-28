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
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util
from postprocess import util as pp_util
from config import rank_args
from rank.remove import get_mean_df
from rank.remove import plot_mean_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()
    color, line, label = pp_util.get_plot_dicts()

    df_loss_list = []
    df_li_loss_list = []

    df_acc_list = []
    df_li_acc_list = []

    df_auc_list = []
    df_li_auc_list = []

    df_loss_rel_list = []
    df_acc_rel_list = []
    df_auc_rel_list = []

    for tree_type in args.tree_type:

        in_dir = os.path.join(args.in_dir,
                              tree_type,
                              f'exp_{exp_hash}',
                              'summary')

        for ckpt in args.ckpt:
            ckpt_dir = os.path.join(in_dir, f'ckpt_{ckpt}')

            # rankings
            fp_loss = os.path.join(ckpt_dir, 'loss_rank.csv')
            fp_li_loss = os.path.join(ckpt_dir, 'loss_rank_li.csv')
            fp_acc = os.path.join(ckpt_dir, 'acc_rank.csv')
            fp_li_acc = os.path.join(ckpt_dir, 'acc_rank_li.csv')
            fp_auc = os.path.join(ckpt_dir, 'auc_rank.csv')
            fp_li_auc = os.path.join(ckpt_dir, 'auc_rank_li.csv')

            assert os.path.exists(fp_loss), f'{fp_loss} does not exist!'
            assert os.path.exists(fp_li_loss), f'{fp_li_loss} does not exist!'
            assert os.path.exists(fp_acc), f'{fp_acc} does not exist!'
            assert os.path.exists(fp_li_acc), f'{fp_li_acc} does not exist!'
            assert os.path.exists(fp_auc), f'{fp_auc} does not exist!'
            assert os.path.exists(fp_auc), f'{fp_auc} doess not exist!'

            df_loss_list.append(pd.read_csv(fp_loss))
            df_li_loss_list.append(pd.read_csv(fp_li_loss))
            df_acc_list.append(pd.read_csv(fp_acc))
            df_li_acc_list.append(pd.read_csv(fp_li_acc))
            df_auc_list.append(pd.read_csv(fp_auc))
            df_li_auc_list.append(pd.read_csv(fp_li_auc))

            # relative performance
            fp_loss_rel = os.path.join(ckpt_dir, 'loss_rel.csv')
            fp_acc_rel = os.path.join(ckpt_dir, 'acc_rel.csv')
            fp_auc_rel = os.path.join(ckpt_dir, 'auc_rel.csv')

            assert os.path.exists(fp_loss_rel), f'{fp_loss_rel} does not exist!'
            assert os.path.exists(fp_acc_rel), f'{fp_acc_rel} does not exist!'
            assert os.path.exists(fp_auc_rel), f'{fp_auc_rel} does not exist!'

            df_loss_rel_list.append(pd.read_csv(fp_loss_rel))
            df_acc_rel_list.append(pd.read_csv(fp_acc_rel))
            df_auc_rel_list.append(pd.read_csv(fp_auc_rel))

    # compile results
    df_loss_all = pd.concat(df_loss_list)
    df_li_loss_all = pd.concat(df_li_loss_list)
    df_acc_all = pd.concat(df_acc_list)
    df_li_acc_all = pd.concat(df_li_acc_list)
    df_auc_all = pd.concat(df_auc_list)
    df_li_auc_all = pd.concat(df_li_auc_list)

    df_loss_rel_all = pd.concat(df_loss_rel_list)
    df_acc_rel_all = pd.concat(df_acc_rel_list)
    df_auc_rel_all = pd.concat(df_auc_rel_list)

    # average among different checkpoints
    group_cols = ['dataset']

    df_loss_all = df_loss_all.groupby(group_cols).mean().reset_index()
    df_li_loss_all = df_li_loss_all.groupby(group_cols).mean().reset_index()
    df_acc_all = df_acc_all.groupby(group_cols).mean().reset_index()
    df_li_acc_all = df_li_acc_all.groupby(group_cols).mean().reset_index()
    df_auc_all = df_auc_all.groupby(group_cols).mean().reset_index()
    df_li_auc_all = df_li_auc_all.groupby(group_cols).mean().reset_index()

    df_loss_rel_all = df_loss_rel_all.groupby(group_cols).mean().reset_index()
    df_acc_rel_all = df_acc_rel_all.groupby(group_cols).mean().reset_index()
    df_auc_rel_all = df_auc_rel_all.groupby(group_cols).mean().reset_index()

    # compute average ranks
    skip_cols = ['dataset', 'tree_type', 'edit_frac']
    remove_cols = ['LeafInfluence', 'LeafRefit']

    df_loss = get_mean_df(df_loss_all, skip_cols=skip_cols, sort='ascending')
    df_li_loss = get_mean_df(df_li_loss_all, skip_cols=skip_cols, sort='ascending')
    df_acc = get_mean_df(df_acc_all, skip_cols=skip_cols, sort='ascending')
    df_li_acc = get_mean_df(df_li_acc_all, skip_cols=skip_cols, sort='ascending')
    df_auc = get_mean_df(df_auc_all, skip_cols=skip_cols, sort='ascending')
    df_li_auc = get_mean_df(df_li_auc_all, skip_cols=skip_cols, sort='ascending')

    df_loss_rel = get_mean_df(df_loss_rel_all, skip_cols=skip_cols + remove_cols, sort='descending', geo_mean=True)
    df_li_loss_rel = get_mean_df(df_loss_rel_all, skip_cols=skip_cols, sort='descending', geo_mean=True)
    df_acc_rel = get_mean_df(df_acc_rel_all, skip_cols=skip_cols + remove_cols, sort='ascending', geo_mean=True)
    df_li_acc_rel = get_mean_df(df_acc_rel_all, skip_cols=skip_cols, sort='ascending', geo_mean=True)
    df_auc_rel = get_mean_df(df_auc_rel_all, skip_cols=skip_cols + remove_cols, sort='ascending', geo_mean=True)
    df_li_auc_rel = get_mean_df(df_auc_rel_all, skip_cols=skip_cols, sort='ascending', geo_mean=True)

    logger.info(f'\nLoss (ranking):\n{df_loss}')
    logger.info(f'\nLoss (ranking-LI):\n{df_li_loss}')
    logger.info(f'\nAcc. (ranking):\n{df_acc}')
    logger.info(f'\nAcc. (ranking-LI):\n{df_li_acc}')
    logger.info(f'\nAUC (ranking):\n{df_auc}')
    logger.info(f'\nAUC (ranking-LI):\n{df_li_auc}')

    logger.info(f'\nLoss (relative):\n{df_loss_rel}')
    logger.info(f'\nLoss (relative-LI):\n{df_li_loss_rel}')
    logger.info(f'\nAcc. (relative):\n{df_acc_rel}')
    logger.info(f'\nAcc. (relative-LI):\n{df_li_acc_rel}')
    logger.info(f'\nAUC (relative):\n{df_auc_rel}')
    logger.info(f'\nAUC (relative-LI):\n{df_li_auc_rel}')

    label_dict = {'Target': 'RandomSL'}

    df_loss = df_loss.rename(index=label_dict)
    df_li_loss = df_li_loss.rename(index=label_dict)
    df_loss_rel = df_loss_rel.rename(index=label_dict)
    df_li_loss_rel = df_li_loss_rel.rename(index=label_dict)

    df_acc = df_acc.rename(index=label_dict)
    df_li_acc = df_li_acc.rename(index=label_dict)
    df_acc_rel = df_acc_rel.rename(index=label_dict)
    df_li_acc_rel = df_li_acc_rel.rename(index=label_dict)

    df_auc = df_auc.rename(index=label_dict)
    df_li_auc = df_li_auc.rename(index=label_dict)
    df_auc_rel = df_auc_rel.rename(index=label_dict)
    df_li_auc_rel = df_li_auc_rel.rename(index=label_dict)

    order = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim', 'SubSample', 'LOO']
    order_li = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim', 'LeafRefit', 'LeafInfluence', 'SubSample', 'LOO']

    if 'RandomSL' in df_loss.index:
        order.append('RandomSL')
        order_li.append('RandomSL')

    df_loss = df_loss.reindex(order)
    df_li_loss = df_li_loss.reindex(order_li)
    df_loss_rel = df_loss_rel.reindex(order)
    df_li_loss_rel = df_li_loss_rel.reindex(order_li)

    df_acc = df_acc.reindex(order)
    df_li_acc = df_li_acc.reindex(order_li)
    df_acc_rel = df_acc_rel.reindex(order)
    df_li_acc_rel = df_li_acc_rel.reindex(order_li)

    df_auc = df_auc.reindex(order)
    df_li_auc = df_li_auc.reindex(order_li)
    df_auc_rel = df_auc_rel.reindex(order)
    df_li_auc_rel = df_li_auc_rel.reindex(order_li)

    logger.info(f'\nSaving results to {out_dir}/...')

    height = 2

    plot_mean_df(df_loss, df_li_loss, out_dir=out_dir, fn='loss_rank', ylabel='Avg. rank', add_height=height)
    plot_mean_df(df_acc, df_li_acc, out_dir=out_dir, fn='acc_rank', ylabel='Avg. rank', add_height=height)
    plot_mean_df(df_auc, df_li_auc, out_dir=out_dir, fn='auc_rank', ylabel='Avg. rank', add_height=height)

    plot_mean_df(df_loss_rel, df_li_loss_rel, out_dir=out_dir, fn='loss_magnitude', yerr=None,
                 ylabel=r'Gmean. loss $\uparrow$' '\n(rel. to Random)', add_height=height)
    plot_mean_df(df_acc_rel, df_li_acc_rel, out_dir=out_dir, fn='acc_magnitude', yerr=None,
                 ylabel=r'Gmean. acc. $\uparrow$' '\n(rel. to Random)', add_height=height)
    plot_mean_df(df_auc_rel, df_li_auc_rel, out_dir=out_dir, fn='auc_magnitude', yerr=None,
                 ylabel=r'Gmean. AUC $\uparrow$' '\n(rel. to Random)', add_height=height)

    pp_util.plot_settings(fontsize=12)

    df_loss.to_csv(os.path.join(out_dir, 'loss_rank.csv'))
    df_li_loss.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'))
    df_acc.to_csv(os.path.join(out_dir, 'acc_rank.csv'))
    df_li_acc.to_csv(os.path.join(out_dir, 'acc_rank_li.csv'))
    df_auc.to_csv(os.path.join(out_dir, 'auc_rank.csv'))
    df_li_auc.to_csv(os.path.join(out_dir, 'auc_rank_li.csv'))

    df_loss_rel.to_csv(os.path.join(out_dir, 'loss_rel.csv'))
    df_li_loss_rel.to_csv(os.path.join(out_dir, 'loss_rel_li.csv'))
    df_acc_rel.to_csv(os.path.join(out_dir, 'acc_rel.csv'))
    df_li_acc_rel.to_csv(os.path.join(out_dir, 'acc_rel_li.csv'))
    df_auc_rel.to_csv(os.path.join(out_dir, 'auc_rel.csv'))
    df_li_auc_rel.to_csv(os.path.join(out_dir, 'auc_rel_li.csv'))

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'edit_frac': args.edit_frac, 'val_frac': args.val_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    assert len(args.tree_type) > 0
    out_dir = os.path.join(args.in_dir,
                           'rank',
                           f'exp_{exp_hash}',
                           '+'.join(args.tree_type))

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'timestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(rank_args.get_label_set_args().parse_args())
