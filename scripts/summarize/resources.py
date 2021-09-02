"""
Aggregate results.
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
import util as pp_util
from experiments import util
from postprocess.leaf_analysis import filter_results


def process(args, out_dir, logger):
    begin = time.time()

    # result containers
    time_rows = []
    time_rows2 = []
    mem_rows = []
    mem_rows2 = []

    color, line, label = pp_util.get_plot_dicts()

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'\n{dataset}')

        args.dataset = dataset

        # collect results
        r1 = {}
        for random_state in args.random_state:
            logger.info(f'random state: {random_state}')

            r1[random_state] = {}

            exp_dir = os.path.join(args.in_dir,
                                   args.dataset,
                                   args.tree_type,
                                   f'random_state_{random_state}')
            res = filter_results(pp_util.get_results(args, args.in_dir, logger, exp_hash='',
                                                     temp_dir=exp_dir, progress_bar=False), args.skip)

            for method, d in res:
                r = {}
                r['mem_GB'] = d['max_rss_MB']  # results were run on Linux
                r['fit_time'] = d['fit_time']
                r['inf_time'] = d['inf_time']
                r['total_time'] = r['fit_time'] + r['inf_time']

                r1[random_state][label[method]] = r

        t = {'dataset': args.dataset}
        t2 = {'dataset': args.dataset}
        m = {'dataset': args.dataset}
        m2 = {'dataset': args.dataset}

        # average over random states
        for method in r1[random_state].keys():

            time_vals = [r1[rs][method]['total_time'] for rs in r1.keys()]
            time_mean = np.mean(time_vals)
            time_std = np.std(time_vals)

            mem_vals = [r1[rs][method]['mem_GB'] for rs in r1.keys()]
            mem_mean = np.mean(mem_vals)
            mem_std = np.std(mem_vals)

            t[f'{method}'] = time_mean
            t2[f'{method}'] = time_std

            m[f'{method}'] = mem_mean
            m2[f'{method}'] = mem_std

        time_rows.append(t)
        time_rows2.append(t2)

        mem_rows.append(m)
        mem_rows2.append(m2)

    # compile results
    t_df = pd.DataFrame(time_rows)
    t2_df = pd.DataFrame(time_rows2)

    m_df = pd.DataFrame(mem_rows)
    m2_df = pd.DataFrame(mem_rows2)

    logger.info(f'\nTime results:\n{t_df}')
    logger.info(f'\nTime std. dev.:\n{t2_df}')

    logger.info(f'\nMemory results:\n{m_df}')
    logger.info(f'\nMemory std. dev.:\n{m2_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    t_df.to_csv(os.path.join(out_dir, 'time.csv'), index=None)
    t2_df.to_csv(os.path.join(out_dir, 'time_std.csv'), index=None)

    m_df.to_csv(os.path.join(out_dir, 'mem.csv'), index=None)
    m2_df.to_csv(os.path.join(out_dir, 'mem_std.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           'summary')

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
    parser.add_argument('--in_dir', type=str, default='temp_resources/')
    parser.add_argument('--out_dir', type=str, default='output/plot/resources/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--dataset_list', type=str, nargs='+',
                        default=['adult', 'bank_marketing', 'bean', 'compas',
                                 'concrete', 'credit_card', 'diabetes', 'energy',
                                 'flight_delays', 'german_credit', 'htru2', 'life',
                                 'msd', 'naval', 'no_show', 'obesity', 'power', 'protein',
                                 'spambase', 'surgical', 'twitter', 'vaccine', 'wine'])
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--strategy', type=str, nargs='+', default=['test_sum', 'self'])
    parser.add_argument('--noise', type=str, default='target')
    parser.add_argument('--noise_frac', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--check_frac', type=float, default=0.1)

    # additional settings
    parser.add_argument('--random_state', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['trex', 'similarity2', 'boostin2', 'leaf_influenceSP', 'subsample', 'loo'])
    parser.add_argument('--skip', type=str, nargs='+', default=[])
    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1, 0])  # LeafInfluence

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

    args = parser.parse_args()
    main(args)
