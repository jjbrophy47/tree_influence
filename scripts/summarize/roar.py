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
from postprocess.leaf_analysis import filter_results
from config import summ_args


def process(args, out_dir, exp_hash, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    n_test = None

    rows = []
    rows2 = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        exp_dir = os.path.join(args.in_dir,
                               dataset,
                               args.tree_type,
                               f'exp_{exp_hash}')

        res_list = pp_util.get_results(args, args.in_dir, exp_dir, logger, progress_bar=False)
        res_list = filter_results(res_list, args.skip)

        row = {'dataset': dataset}
        row2 = {'dataset': dataset}

        for j, (method, res) in enumerate(res_list):

            # sanity check
            if j == 0:
                n_test = res['loss'].shape[0]

            else:
                temp = res['loss'].shape[0]
                assert n_test == temp, f'Inconsistent no. test: {temp:,} != {n_test:,}'

            loss_mean = res['loss'].mean(axis=0)[args.ckpt]
            loss_sem = sem(res['loss'], axis=0)[args.ckpt]

            row['remove_frac'] = res['remove_frac'][args.ckpt]
            row[f'{label[method]}'] = loss_mean

            row2['remove_frac'] = row['remove_frac']
            row2[f'{label[method]}'] = loss_sem

        rows.append(row)
        rows2.append(row2)

    df = pd.DataFrame(rows)
    df2 = pd.DataFrame(rows2)
    logger.info(f'\nLoss:\n{df}')

    logger.info(f'\nSaving results to {out_dir}...')

    df.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)
    df2.to_csv(os.path.join(out_dir, 'loss_sem.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
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

    process(args, out_dir, exp_hash, logger)


if __name__ == '__main__':
    main(summ_args.get_roar_args().parse_args())
