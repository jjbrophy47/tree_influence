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
from summarize.roar import get_rank_df


def process(args, exp_hash, out_dir, logger):
    color, line, label = pp_util.get_plot_dicts()

    fd_d_list = []
    loss_d_list = []
    acc_d_list = []
    auc_d_list = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        r1 = {}

        for random_state in range(1, args.n_repeat + 1):
            r1[random_state] = {}

            for strategy in args.strategy:

                exp_dir = os.path.join(args.in_dir,
                                       dataset,
                                       args.tree_type,
                                       f'exp_{exp_hash}',
                                       strategy,
                                       f'random_state_{random_state}')

                res_list = pp_util.get_results(args, exp_dir, logger, progress_bar=False)
                res_list = pp_util.filter_results(res_list, args.skip)

                for method, res in res_list:

                    res_simple = {}
                    res_simple['frac_detected'] = res['frac_detected'][args.ckpt]
                    res_simple['loss'] = res['loss'][args.ckpt]
                    res_simple['acc'] = res['acc'][args.ckpt]
                    res_simple['auc'] = res['auc'][args.ckpt]

                    name = f'{label[method]}_{strategy}'

                    r1[random_state][name] = res_simple

        # average over random states
        temp_dict = {'dataset': dataset, 'noise_frac': args.noise_frac,
                     'check_frac': args.check_frac[args.ckpt], 'tree_type': args.tree_type}
        fd_d = temp_dict.copy()
        loss_d = temp_dict.copy()
        acc_d = temp_dict.copy()
        auc_d = temp_dict.copy()

        for name in r1[random_state].keys():
            try:
                fd_d[name] = np.mean([r1[rs][name]['frac_detected'] for rs in r1.keys()])
                loss_d[name] = np.mean([r1[rs][name]['loss'] for rs in r1.keys()])
                acc_d[name] = np.mean([r1[rs][name]['acc'] for rs in r1.keys()])
                auc_d[name] = np.mean([r1[rs][name]['auc'] for rs in r1.keys()])
            except:
                logger.info(f'\tfailed averaging random states for {name}')

        fd_d_list.append(fd_d)
        loss_d_list.append(loss_d)
        acc_d_list.append(acc_d)
        auc_d_list.append(auc_d)

    # organize results
    fd_df = pd.DataFrame(fd_d_list).replace(-1, np.nan)
    loss_df = pd.DataFrame(loss_d_list).replace(-1, np.nan)
    acc_df = pd.DataFrame(acc_d_list).replace(-1, np.nan)
    auc_df = pd.DataFrame(auc_d_list).replace(-1, np.nan)

    logger.info(f'\n\nFrac. detected:\n{fd_df}')
    logger.info(f'\n\nLoss:\n{loss_df}')
    logger.info(f'\n\nAcc.:\n{acc_df}')
    logger.info(f'\n\nAUC:\n{auc_df}')

    # compute ranks
    skip_cols = ['dataset', 'tree_type', 'noise_frac', 'check_frac']
    remove_cols = ['Leaf Inf._test_sum', 'Leaf Refit_test_sum']

    rank_df_fd = get_rank_df(fd_df, skip_cols=skip_cols, remove_cols=remove_cols)
    rank_li_df_fd = get_rank_df(fd_df[~pd.isna(fd_df['Leaf Inf._test_sum'])], skip_cols=skip_cols)
    logger.info(f'\nFrac. detected ranking:\n{rank_df_fd}')
    logger.info(f'\nFrac. detected ranking (w/ leafinf):\n{rank_li_df_fd}')

    rank_df_loss = get_rank_df(loss_df, skip_cols=skip_cols, remove_cols=remove_cols, ascending=True)
    rank_li_df_loss = get_rank_df(loss_df[~pd.isna(loss_df['Leaf Inf._test_sum'])], skip_cols=skip_cols, ascending=True)
    logger.info(f'\nLoss ranking:\n{rank_df_loss}')
    logger.info(f'\nLoss ranking (w/ leafinf):\n{rank_li_df_loss}')

    rank_df_acc = get_rank_df(acc_df, skip_cols=skip_cols, remove_cols=remove_cols)
    rank_li_df_acc = get_rank_df(acc_df[~pd.isna(acc_df['Leaf Inf._test_sum'])], skip_cols=skip_cols)
    logger.info(f'\nAcc. ranking:\n{rank_df_acc}')
    logger.info(f'\nAcc. ranking (w/ leafinf):\n{rank_li_df_acc}')

    rank_df_auc = get_rank_df(auc_df, skip_cols=skip_cols, remove_cols=remove_cols)
    rank_li_df_auc = get_rank_df(auc_df[~pd.isna(auc_df['Leaf Inf._test_sum'])], skip_cols=skip_cols)
    logger.info(f'\nAUC ranking:\n{rank_df_auc}')
    logger.info(f'\nAUC ranking (w/ leafinf):\n{rank_li_df_auc}')

    # save results
    logger.info(f'\nSaving results to {out_dir}...')

    fd_df.to_csv(os.path.join(out_dir, 'frac_detected.csv'), index=None)
    loss_df.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)
    acc_df.to_csv(os.path.join(out_dir, 'acc.csv'), index=None)
    auc_df.to_csv(os.path.join(out_dir, 'auc.csv'), index=None)

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

    exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
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
    main(summ_args.get_noise_args().parse_args())
