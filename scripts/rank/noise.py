"""
Aggregate results.

TODO: Average over multiple noise fracs.?
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
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util
from postprocess import util as pp_util
from config import rank_args
from rank.roar import get_mean_rank_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()

    # get dataset
    color, line, label = pp_util.get_plot_dicts()

    df_fd_list = []
    df_li_fd_list = []

    df_loss_list = []
    df_li_loss_list = []

    df_acc_list = []
    df_li_acc_list = []

    df_auc_list = []
    df_li_auc_list = []

    for tree_type in args.tree_type:

        in_dir = os.path.join(args.in_dir,
                              tree_type,
                              f'exp_{exp_hash}',
                              'summary')

        for ckpt in args.ckpt:
            ckpt_dir = os.path.join(in_dir, f'ckpt_{ckpt}')

            # define paths
            fp_fd = os.path.join(ckpt_dir, 'frac_detected_rank.csv')
            fp_li_fd = os.path.join(ckpt_dir, 'frac_detected_rank_li.csv')
            fp_loss = os.path.join(ckpt_dir, 'loss_rank.csv')
            fp_li_loss = os.path.join(ckpt_dir, 'loss_rank_li.csv')
            fp_acc = os.path.join(ckpt_dir, 'acc_rank.csv')
            fp_li_acc = os.path.join(ckpt_dir, 'acc_rank_li.csv')
            fp_auc = os.path.join(ckpt_dir, 'auc_rank.csv')
            fp_li_auc = os.path.join(ckpt_dir, 'auc_rank_li.csv')

            # check paths
            assert os.path.exists(fp_fd), f'{fp_fd} does not exist!'
            assert os.path.exists(fp_li_fd), f'{fp_li_fd} does not exist!'
            assert os.path.exists(fp_loss), f'{fp_loss} does not exist!'
            assert os.path.exists(fp_li_loss), f'{fp_li_loss} does not exist!'
            assert os.path.exists(fp_acc), f'{fp_acc} does not exist!'
            assert os.path.exists(fp_li_acc), f'{fp_li_acc} does not exist!'
            assert os.path.exists(fp_auc), f'{fp_auc} does not exist!'
            assert os.path.exists(fp_auc), f'{fp_auc} doess not exist!'

            # read results
            df_fd_list.append(pd.read_csv(fp_fd))
            df_li_fd_list.append(pd.read_csv(fp_li_fd))
            df_loss_list.append(pd.read_csv(fp_loss))
            df_li_loss_list.append(pd.read_csv(fp_li_loss))
            df_acc_list.append(pd.read_csv(fp_acc))
            df_li_acc_list.append(pd.read_csv(fp_li_acc))
            df_auc_list.append(pd.read_csv(fp_auc))
            df_li_auc_list.append(pd.read_csv(fp_li_auc))

    # compile results
    df_fd_all = pd.concat(df_fd_list)
    df_li_fd_all = pd.concat(df_li_fd_list)
    df_loss_all = pd.concat(df_loss_list)
    df_li_loss_all = pd.concat(df_li_loss_list)
    df_acc_all = pd.concat(df_acc_list)
    df_li_acc_all = pd.concat(df_li_acc_list)
    df_auc_all = pd.concat(df_auc_list)
    df_li_auc_all = pd.concat(df_li_auc_list)

    # average ranks among different checkpoints
    group_cols = ['dataset', 'tree_type', 'noise_frac']

    df_fd_all = df_fd_all.groupby(group_cols).mean().reset_index()
    df_li_fd_all = df_li_fd_all.groupby(group_cols).mean().reset_index()
    df_loss_all = df_loss_all.groupby(group_cols).mean().reset_index()
    df_li_loss_all = df_li_loss_all.groupby(group_cols).mean().reset_index()
    df_acc_all = df_acc_all.groupby(group_cols).mean().reset_index()
    df_li_acc_all = df_li_acc_all.groupby(group_cols).mean().reset_index()
    df_auc_all = df_auc_all.groupby(group_cols).mean().reset_index()
    df_li_auc_all = df_li_auc_all.groupby(group_cols).mean().reset_index()

    # compute average ranks
    skip_cols = ['dataset', 'tree_type', 'noise_frac', 'check_frac']

    df_fd = get_mean_rank_df(df_fd_all, skip_cols=skip_cols, sort='ascending')
    df_li_fd = get_mean_rank_df(df_li_fd_all, skip_cols=skip_cols, sort='ascending')
    df_loss = get_mean_rank_df(df_loss_all, skip_cols=skip_cols, sort='ascending')
    df_li_loss = get_mean_rank_df(df_li_loss_all, skip_cols=skip_cols, sort='ascending')
    df_acc = get_mean_rank_df(df_acc_all, skip_cols=skip_cols, sort='ascending')
    df_li_acc = get_mean_rank_df(df_li_acc_all, skip_cols=skip_cols, sort='ascending')
    df_auc = get_mean_rank_df(df_auc_all, skip_cols=skip_cols, sort='ascending')
    df_li_auc = get_mean_rank_df(df_li_auc_all, skip_cols=skip_cols, sort='ascending')

    # plot
    n_fd_datasets = len(df_fd_all['dataset'].unique())
    n_li_fd_datasets = len(df_li_fd_all['dataset'].unique())
    n_loss_datasets = len(df_loss_all['dataset'].unique())
    n_li_loss_datasets = len(df_li_loss_all['dataset'].unique())
    n_acc_datasets = len(df_acc_all['dataset'].unique())
    n_li_acc_datasets = len(df_li_acc_all['dataset'].unique())
    n_auc_datasets = len(df_auc_all['dataset'].unique())
    n_li_auc_datasets = len(df_li_auc_all['dataset'].unique())

    fig, axs = plt.subplots(2, 4, figsize=(18, 8))

    ax = axs[0][0]
    df_fd.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
               title=f'Frac. Detected ({n_loss_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1][0]
    df_li_fd.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                  title=f'w/ LeafInf ({n_li_loss_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[0][1]
    df_loss.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                 title=f'Loss ({n_loss_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1][1]
    df_li_loss.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                    title=f'w/ LeafInf ({n_li_loss_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[0][2]
    df_acc.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                title=f'Accuracy ({n_acc_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1][2]
    df_li_acc.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                   title=f'w/ LeafInf ({n_li_acc_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[0][3]
    df_auc.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                title=f'AUC ({n_auc_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1][3]
    df_li_auc.plot(kind='bar', y='mean', yerr='sem', ax=ax, rot=45, legend=None, capsize=3,
                   title=f'w/ LeafInf ({n_li_auc_datasets} datasets)', ylabel='Avg. rank', xlabel='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rank.png'), bbox_inches='tight')
    plt.show()

    df_fd.to_csv(os.path.join(out_dir, 'fd_rank.csv'))
    df_li_fd.to_csv(os.path.join(out_dir, 'fd_rank_li.csv'))
    df_loss.to_csv(os.path.join(out_dir, 'loss_rank.csv'))
    df_li_loss.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'))
    df_acc.to_csv(os.path.join(out_dir, 'acc_rank.csv'))
    df_li_acc.to_csv(os.path.join(out_dir, 'acc_rank_li.csv'))
    df_auc.to_csv(os.path.join(out_dir, 'auc_rank.csv'))
    df_li_auc.to_csv(os.path.join(out_dir, 'auc_rank_li.csv'))

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
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
    main(rank_args.get_noise_args().parse_args())
