"""
Aggregate results.
"""
import os
import sys
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
    color, line, label = pp_util.get_plot_dicts()

    rows_fd = []
    rows_loss = []
    rows_acc = []
    rows_auc = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        row_fd = {'dataset': dataset, 'tree_type': args.tree_type}
        row_loss = row_fd.copy()
        row_acc = row_fd.copy()
        row_auc = row_fd.copy()

        for strategy in args.strategy:

            exp_dir = os.path.join(args.in_dir,
                                   dataset,
                                   args.tree_type,
                                   f'exp_{exp_hash}',
                                   strategy)

            res_list = pp_util.get_results(args, exp_dir, logger, progress_bar=False)
            res_list = pp_util.filter_results(res_list, args.skip)

            for method, res in res_list:
                name = f'{label[method]}_{strategy}'

                row_fd['check_frac'] = res['check_frac'][args.ckpt]
                row_fd[name] = res['frac_detected'][args.ckpt]

                row_loss['check_frac'] = res['check_frac'][args.ckpt]
                row_loss[name] = res['loss'][args.ckpt]

                row_acc['check_frac'] = res['check_frac'][args.ckpt]
                row_acc[name] = res['acc'][args.ckpt]

                row_auc['check_frac'] = res['check_frac'][args.ckpt]
                row_auc[name] = res['auc'][args.ckpt]

        rows_fd.append(row_fd)
        rows_loss.append(row_loss)
        rows_acc.append(row_acc)
        rows_auc.append(row_auc)

    # organize results
    df_fd_raw = pd.DataFrame(rows_fd).replace(-1, np.nan)
    df_loss_raw = pd.DataFrame(rows_loss).replace(-1, np.nan)
    df_acc_raw = pd.DataFrame(rows_acc).replace(-1, np.nan)
    df_auc_raw = pd.DataFrame(rows_auc).replace(-1, np.nan)

    # # skip BoostIn_self
    # df_fd_raw = df_fd_raw.drop(['BoostIn_self'], axis=1)
    # df_loss_raw = df_loss_raw.drop(['BoostIn_self'], axis=1)
    # df_acc_raw = df_acc_raw.drop(['BoostIn_self'], axis=1)
    # df_auc_raw = df_auc_raw.drop(['BoostIn_self'], axis=1)

    # drop rows with missing values
    skip_cols = ['dataset', 'tree_type', 'noise_frac', 'check_frac']
    remove_cols = ['LeafInfluence_test_sum', 'LeafRefit_test_sum']

    cols = [x for x in df_fd_raw.columns if x not in skip_cols + remove_cols]

    df_fd = df_fd_raw.dropna(subset=cols)
    df_loss = df_loss_raw.dropna(subset=cols)
    df_acc = df_acc_raw.dropna(subset=cols)
    df_auc = df_auc_raw.dropna(subset=cols)

    logger.info(f'\n\nFrac. detected:\n{df_fd}')
    logger.info(f'\n\nLoss:\n{df_loss}')
    logger.info(f'\n\nAcc.:\n{df_acc}')
    logger.info(f'\n\nAUC:\n{df_auc}')

    # compute relative performance and rankings
    skip_cols = ['dataset', 'tree_type', 'noise_frac', 'check_frac']
    remove_cols = ['LeafInfluence_test_sum', 'LeafRefit_test_sum']
    ref_col = 'Random_test_sum'

    # relative performance
    rel_df_fd = get_relative_df(df_fd, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    rel_df_loss = get_relative_df(df_loss, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    rel_df_acc = get_relative_df(df_acc, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])
    rel_df_auc = get_relative_df(df_auc, ref_col=ref_col, skip_cols=skip_cols, remove_cols=[ref_col])

    logger.info(f'\nFrac. detected (relative):\n{rel_df_fd}')
    logger.info(f'\nLoss (relative):\n{rel_df_loss}')
    logger.info(f'\nAccuracy (relative):\n{rel_df_acc}')
    logger.info(f'\nAUC (relative):\n{rel_df_auc}')

    # rankings
    rank_df_fd = get_rank_df(df_fd, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col])
    rank_df_loss = get_rank_df(df_loss, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col], ascending=True)
    rank_df_acc = get_rank_df(df_acc, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col])
    rank_df_auc = get_rank_df(df_auc, skip_cols=skip_cols, remove_cols=remove_cols + [ref_col])

    rank_li_df_fd = get_rank_df(df_fd[~pd.isna(df_fd['LeafInfluence_test_sum'])], skip_cols, [ref_col])
    rank_li_df_loss = get_rank_df(df_loss[~pd.isna(df_loss['LeafInfluence_test_sum'])],
                                  skip_cols, [ref_col], ascending=True)
    rank_li_df_acc = get_rank_df(df_acc[~pd.isna(df_acc['LeafInfluence_test_sum'])], skip_cols, [ref_col])
    rank_li_df_auc = get_rank_df(df_auc[~pd.isna(df_auc['LeafInfluence_test_sum'])], skip_cols, [ref_col])

    logger.info(f'\nFrac. detected ranking:\n{rank_df_fd}')
    logger.info(f'\nFrac. detected ranking (w/ leafinf):\n{rank_li_df_fd}')

    logger.info(f'\nLoss ranking:\n{rank_df_loss}')
    logger.info(f'\nLoss ranking (w/ leafinf):\n{rank_li_df_loss}')

    logger.info(f'\nAUC ranking:\n{rank_df_auc}')
    logger.info(f'\nAUC ranking (w/ leafinf):\n{rank_li_df_auc}')

    logger.info(f'\nAcc. ranking:\n{rank_df_acc}')
    logger.info(f'\nAcc. ranking (w/ leafinf):\n{rank_li_df_acc}')

    # save results
    logger.info(f'\nSaving results to {out_dir}...')

    df_fd_raw.to_csv(os.path.join(out_dir, 'frac_detected.csv'), index=None)
    df_loss_raw.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)
    df_acc_raw.to_csv(os.path.join(out_dir, 'acc.csv'), index=None)
    df_auc_raw.to_csv(os.path.join(out_dir, 'auc.csv'), index=None)

    rel_df_fd.to_csv(os.path.join(out_dir, 'fd_rel.csv'), index=None)
    rel_df_loss.to_csv(os.path.join(out_dir, 'loss_rel.csv'), index=None)
    rel_df_acc.to_csv(os.path.join(out_dir, 'acc_rel.csv'), index=None)
    rel_df_auc.to_csv(os.path.join(out_dir, 'auc_rel.csv'), index=None)

    rank_df_fd.to_csv(os.path.join(out_dir, 'frac_detected_rank.csv'), index=None)
    rank_df_loss.to_csv(os.path.join(out_dir, 'loss_rank.csv'), index=None)
    rank_df_acc.to_csv(os.path.join(out_dir, 'acc_rank.csv'), index=None)
    rank_df_auc.to_csv(os.path.join(out_dir, 'auc_rank.csv'), index=None)

    rank_li_df_fd.to_csv(os.path.join(out_dir, 'frac_detected_rank_li.csv'), index=None)
    rank_li_df_loss.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'), index=None)
    rank_li_df_acc.to_csv(os.path.join(out_dir, 'acc_rank_li.csv'), index=None)
    rank_li_df_auc.to_csv(os.path.join(out_dir, 'auc_rank_li.csv'), index=None)


def main(args):

    args.method += ['loss']

    exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac,
                'check_frac': args.check_frac}
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
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_noise_set_args().parse_args())
