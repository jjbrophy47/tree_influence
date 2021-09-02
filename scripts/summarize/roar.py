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
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util as pp_util
from experiments import util
from postprocess.leaf_analysis import filter_results


def process(args, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    n_test = None

    rows = []
    rows2 = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        args.dataset = dataset
        res_list = filter_results(pp_util.get_results(args, args.in_dir,
                                                      logger, progress_bar=False), args.skip)

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
    logger.info(f'\nLoss w/ std. error:\n{df2}')

    logger.info(f'\nSaving results to {out_dir}...')

    df.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)
    df2.to_csv(os.path.join(out_dir, 'loss_sem.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
                'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           'summary',
                           f'ckpt_{args.ckpt}')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_influence/')
    parser.add_argument('--out_dir', type=str, default='output/plot/roar/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--dataset_list', type=str, nargs='+',
                        default=['adult', 'bank_marketing', 'bean', 'compas',
                                 'concrete', 'credit_card', 'diabetes', 'energy',
                                 'flight_delays', 'german_credit', 'htru2', 'life',
                                 'msd', 'naval', 'no_show', 'obesity', 'power', 'protein',
                                 'spambase', 'surgical', 'twitter', 'vaccine', 'wine'])
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
    parser.add_argument('--n_ckpt', type=int, default=50)

    # additional settings
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['trex', 'similarity', 'boostin2', 'leaf_influenceSP',
                                 'subsample', 'loo', 'target', 'random'])
    parser.add_argument('--skip', type=str, nargs='+',
                        default=['minority', 'loss'])
    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1])  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default=['dot_prod'])  # Similarity

    parser.add_argument('--kernel', type=str, nargs='+', default=['lpw'])  # Trex & Similarity
    parser.add_argument('--target', type=str, nargs='+', default=['actual'])  # Trex
    parser.add_argument('--lmbd', type=float, nargs='+', default=[0.003])  # Trex
    parser.add_argument('--n_epoch', type=str, nargs='+', default=[3000])  # Trex

    parser.add_argument('--trunc_frac', type=float, nargs='+', default=[0.25])  # DShap
    parser.add_argument('--check_every', type=int, nargs='+', default=[100])  # DShap

    parser.add_argument('--sub_frac', type=float, nargs='+', default=[0.7])  # SubSample
    parser.add_argument('--n_iter', type=int, nargs='+', default=[4000])  # SubSample

    parser.add_argument('--global_op', type=str, nargs='+', default=['self', 'expected'])  # TREX, LOO, DShap

    # result settings
    parser.add_argument('--ckpt', type=int, default=5)  # 0.5%

    args = parser.parse_args()
    main(args)
