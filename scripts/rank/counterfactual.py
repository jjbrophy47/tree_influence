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


def process(args, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    rows = []
    rows2 = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, dataset)

        # get experiment directory
        exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac,
                    'n_ckpt': args.n_ckpt, 'step_size': args.step_size}
        exp_hash = exp_util.dict_to_hash(exp_dict)

        exp_dir = os.path.join(args.in_dir,
                               dataset,
                               args.tree_type,
                               f'exp_{exp_hash}')

        res_list = pp_util.get_results(args, args.in_dir, exp_dir, logger, progress_bar=False)
        res_list = pp_util.filter_results(res_list, args.skip)

        row = {'dataset': dataset}
        row2 = {'dataset': dataset}
        
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

    logger.info(f'\nFrac. edit results:\n{df}')

    logger.info(f'\nSaving results to {out_dir}...')

    df.to_csv(os.path.join(out_dir, 'frac_edits.csv'), index=None)
    df2.to_csv(os.path.join(out_dir, 'frac_edits_sem.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           'summary')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_counterfactual_args().parse_args())
