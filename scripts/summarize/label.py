"""
Aggregate results and organize them into one dict.
"""
import os
import sys
import time
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import gmean
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import summ_args
from remove import get_rank_df
from remove import get_relative_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    n_test = None

    rows = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        exp_dir = os.path.join(args.in_dir,
                               dataset,
                               args.tree_type,
                               f'exp_{exp_hash}')

        res_list = pp_util.get_results(args, exp_dir, logger, progress_bar=False)
        res_list = pp_util.filter_results(res_list, args.skip)

        row = {'dataset': dataset, 'tree_type': args.tree_type}

        for j, (method, res) in enumerate(res_list):

            # sanity check
            if j == 0:
                n_test = res['loss'].shape[0]

            else:
                temp = res['loss'].shape[0]
                assert n_test == temp, f'Inconsistent no. test: {temp:,} != {n_test:,}'

            loss_mean = res['loss'].mean(axis=0)[args.ckpt]

            row['edit_frac'] = res['edit_frac'][args.ckpt]
            row[f'{label[method]}'] = loss_mean

        rows.append(row)

    df = pd.DataFrame(rows)

    # drop rows with missing values
    skip_cols = ['dataset', 'tree_type', 'edit_frac']
    remove_cols = ['LeafInfluence', 'LeafRefit']

    cols = [x for x in df.columns if x not in skip_cols + remove_cols]

    df = df.dropna(subset=cols)

    logger.info(f'\nLoss:\n{df}')

    # compute relative peformance and rankings
    skip_cols = ['dataset', 'tree_type', 'edit_frac']

    # relative performance
    df_rel = get_relative_df(df, ref_col='Random', skip_cols=skip_cols)
    logger.info(f'\nLoss (relative increase):\n{df_rel}')

    # rank
    rank_df = get_rank_df(df, skip_cols=skip_cols, remove_cols=['LeafInfluence', 'LeafRefit'])
    rank_li_df = get_rank_df(df[~pd.isna(df['LeafInfluence'])], skip_cols=skip_cols)
    logger.info(f'\nLoss ranking:\n{rank_df}')
    logger.info(f'\nLoss ranking (w/ leafinf):\n{rank_li_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    df.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)

    df_rel.to_csv(os.path.join(out_dir, 'loss_rel.csv'), index=None)

    rank_df.to_csv(os.path.join(out_dir, 'loss_rank.csv'), index=None)
    rank_li_df.to_csv(os.path.join(out_dir, 'loss_rank_li.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'n_test': args.n_test, 'edit_frac': args.edit_frac}
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
    logger.info(datetime.now())

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_label_args().parse_args())
