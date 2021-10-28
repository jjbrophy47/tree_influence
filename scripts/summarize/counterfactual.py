"""
Summarize counerfactual results.
"""
import os
import sys
import time
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.metrics import log_loss

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import summ_args
from summarize.roar import get_rank_df


def process(args, exp_hash, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    rows = []
    rows2 = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, dataset)

        exp_dir = os.path.join(args.in_dir,
                               dataset,
                               args.tree_type,
                               f'exp_{exp_hash}')

        res_list = pp_util.get_results(args, exp_dir, logger, progress_bar=False)
        res_list = pp_util.filter_results(res_list, args.skip)

        row = {'dataset': dataset, 'tree_type': args.tree_type}
        row2 = {'dataset': dataset, 'tree_type': args.tree_type}
        
        for method, res in res_list:

            df = res['df']
            n_correct = len(df[df['status'] == 'success']) + len(df[df['status'] == 'fail'])

            # result containers
            frac_edits = np.full(n_correct, 0.1, dtype=np.float32)
            n_edits = np.full(n_correct, int(X_train.shape[0] * 0.1), dtype=np.float32)

            temp_df = df[df['status'] == 'success']

            if len(temp_df) > 0:
                frac_edits[:len(temp_df)] = temp_df['frac_edits'].values
                n_edits[:len(temp_df)] = temp_df['n_edits'].values

            row[label[method]] = np.mean(frac_edits)
            row2[label[method]] = sem(frac_edits)

        rows.append(row)
        rows2.append(row2)

    df = pd.DataFrame(rows)
    df2 = pd.DataFrame(rows2)

    # drop rows in which CB did not finish ROAR experiment
    remove_datasets = ['adult', 'bank_marketing', 'diabetes', 'flight_delays', 'htru2',
                       'no_show', 'obesity', 'protein', 'twitter', 'vaccine']

    df = df[~df['dataset'].isin(remove_datasets)]
    df2 = df2[~df2['dataset'].isin(remove_datasets)]

    logger.info(f'\nFrac. edit results:\n{df}')

    # compute ranks
    skip_cols = ['dataset', 'tree_type']

    rank_df = get_rank_df(df, skip_cols=skip_cols, remove_cols=['LeafInfluence', 'LeafRefit'], ascending=True)
    rank_li_df = get_rank_df(df[~pd.isna(df['LeafInfluence'])], skip_cols=skip_cols, ascending=True)
    logger.info(f'\nFrac. edit ranking:\n{rank_df}')
    logger.info(f'\nFrac. edit ranking (w/ leafinf):\n{rank_li_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    df.to_csv(os.path.join(out_dir, 'frac_edits.csv'), index=None)
    df2.to_csv(os.path.join(out_dir, 'frac_edits_sem.csv'), index=None)

    rank_df.to_csv(os.path.join(out_dir, 'frac_edits_rank.csv'), index=None)
    rank_li_df.to_csv(os.path.join(out_dir, 'frac_edits_rank_li.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    if args.tree_type == 'cb':
        args.step_size = 100

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'step_size': args.step_size}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           'summary')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, exp_hash, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_counterfactual_args().parse_args())
