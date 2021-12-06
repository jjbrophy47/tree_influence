"""
Summarize results across all datasets.
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
from config import summ_args
from remove import get_rank_df
from remove import get_relative_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()
    color, line, label = pp_util.get_plot_dicts()

    rows_loss = []
    rows_acc = []
    rows_auc = []

    for dataset in args.dataset_list:

        exp_dir = os.path.join(args.in_dir,
                               dataset,
                               args.tree_type,
                               f'exp_{exp_hash}')

        res_list = pp_util.get_results(args, exp_dir, logger, progress_bar=False)
        res_list = pp_util.filter_results(res_list, args.skip)

        row_loss = {'dataset': dataset, 'tree_type': args.tree_type}
        row_acc = row_loss.copy()
        row_auc = row_loss.copy()

        for method, res in res_list:

            row_loss['edit_frac'] = res['edit_frac'][args.ckpt]
            row_loss[f'{label[method]}'] = res['loss'][args.ckpt]

            row_acc['edit_frac'] = res['edit_frac'][args.ckpt]
            row_acc[f'{label[method]}'] = res['acc'][args.ckpt]

            row_auc['edit_frac'] = res['edit_frac'][args.ckpt]
            row_auc[f'{label[method]}'] = res['auc'][args.ckpt]

        rows_loss.append(row_loss)
        rows_acc.append(row_acc)
        rows_auc.append(row_auc)

    df_loss_raw = pd.DataFrame(rows_loss).replace(-1, np.nan)
    df_acc_raw = pd.DataFrame(rows_acc).replace(-1, np.nan)
    df_auc_raw = pd.DataFrame(rows_auc).replace(-1, np.nan)

    # drop rows with missing values
    skip_cols = ['dataset', 'tree_type', 'edit_frac']
    remove_cols = ['LeafInfluence', 'LeafRefit']

    cols = [x for x in df_loss_raw.columns if x not in skip_cols + remove_cols]

    df_loss = df_loss_raw.dropna(subset=cols)
    df_acc = df_acc_raw.dropna(subset=cols)
    df_auc = df_auc_raw.dropna(subset=cols)

    logger.info(f'\nLoss:\n{df_loss}')
    logger.info(f'\nAccuracy:\n{df_acc}')
    logger.info(f'\nAUC:\n{df_auc}')

    # compute relative performance and rankings
    skip_cols = ['dataset', 'tree_type', 'edit_frac']
    remove_cols = ['LeafInfluence', 'LeafRefit']
    ref_col = 'Random'

    # relative performance
    rel_df_loss = get_relative_df(df_loss, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    rel_df_acc = get_relative_df(df_acc, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    rel_df_auc = get_relative_df(df_auc, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])

    # rel_li_df_loss = get_relative_df(df_loss, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    # rel_li_df_acc = get_relative_df(df_acc, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    # rel_li_df_auc = get_relative_df(df_auc, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])

    logger.info(f'\nLoss (relative):\n{rel_df_loss}')
    # logger.info(f'\nLoss (LI-relative):\n{rel_li_df_loss}')
    logger.info(f'\nAccuracy (relative):\n{rel_df_acc}')
    # logger.info(f'\nAccuracy (LI-relative):\n{rel_li_df_acc}')
    logger.info(f'\nAUC (relative):\n{rel_df_auc}')
    # logger.info(f'\nAUC (LI-relative):\n{rel_li_df_auc}')

    # rankings
    rank_df_loss = get_rank_df(df_loss, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col])
    rank_df_acc = get_rank_df(df_acc, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col], ascending=True)
    rank_df_auc = get_rank_df(df_auc, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col], ascending=True)

    rank_li_df_loss = get_rank_df(df_loss[~pd.isna(df_loss['LeafInfluence'])], skip_cols, [ref_col])
    rank_li_df_acc = get_rank_df(df_acc[~pd.isna(df_acc['LeafInfluence'])], skip_cols, [ref_col], ascending=True)
    rank_li_df_auc = get_rank_df(df_auc[~pd.isna(df_auc['LeafInfluence'])], skip_cols, [ref_col], ascending=True)

    logger.info(f'\nLoss ranking:\n{rank_df_loss}')
    logger.info(f'\nLoss ranking (w/ leafinf):\n{rank_li_df_loss}')

    logger.info(f'\nAcc. ranking:\n{rank_df_acc}')
    logger.info(f'\nAcc. ranking (w/ leafinf):\n{rank_li_df_acc}')

    logger.info(f'\nAUC ranking:\n{rank_df_auc}')
    logger.info(f'\nAUC ranking (w/ leafinf):\n{rank_li_df_auc}')

    logger.info(f'\nSaving results to {out_dir}/...')

    df_loss_raw.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)
    df_acc_raw.to_csv(os.path.join(out_dir, 'acc.csv'), index=None)
    df_auc_raw.to_csv(os.path.join(out_dir, 'auc.csv'), index=None)

    rel_df_loss.to_csv(os.path.join(out_dir, 'loss_rel.csv'), index=None)
    rel_df_acc.to_csv(os.path.join(out_dir, 'acc_rel.csv'), index=None)
    rel_df_auc.to_csv(os.path.join(out_dir, 'auc_rel.csv'), index=None)

    # rel_li_df_loss.to_csv(os.path.join(out_dir, 'loss_rel_li.csv'), index=None)
    # rel_li_df_acc.to_csv(os.path.join(out_dir, 'acc_rel_li.csv'), index=None)
    # rel_li_df_auc.to_csv(os.path.join(out_dir, 'auc_rel_li.csv'), index=None)

    rank_df_loss.to_csv(os.path.join(out_dir, 'loss_rank.csv'), index=None)
    rank_df_acc.to_csv(os.path.join(out_dir, 'acc_rank.csv'), index=None)
    rank_df_auc.to_csv(os.path.join(out_dir, 'auc_rank.csv'), index=None)

    rank_li_df_loss.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'), index=None)
    rank_li_df_acc.to_csv(os.path.join(out_dir, 'acc_rank_li.csv'), index=None)
    rank_li_df_auc.to_csv(os.path.join(out_dir, 'auc_rank_li.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'edit_frac': args.edit_frac, 'val_frac': args.val_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           'summary',
                           f'ckpt_{args.ckpt}')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'timestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_label_set_args().parse_args())
