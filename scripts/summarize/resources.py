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
from postprocess import util as pp_util
from experiments import util as exp_util
from config import summ_args


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
        for random_state in range(1, args.n_repeat + 1):
            logger.info(f'random state: {random_state}')

            r1[random_state] = {}

            exp_dir = os.path.join(args.in_dir,
                                   args.dataset,
                                   args.tree_type,
                                   f'random_state_{random_state}')
            res_list = pp_util.get_results(args, args.in_dir, exp_dir, logger, progress_bar=False)
            res_list = pp_util.filter_results(res_list, args.skip)

            for method, d in res_list:
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
            is_missing = False

            try:
                time_vals = [r1[rs][method]['total_time'] for rs in r1.keys()]
                time_mean = np.mean(time_vals)
                time_std = np.std(time_vals)

                mem_vals = [r1[rs][method]['mem_GB'] for rs in r1.keys()]
                mem_mean = np.mean(mem_vals)
                mem_std = np.std(mem_vals)

            except:
                is_missing = True
                print(f'partially missing: {method}')

                t[f'{method}'] = np.nan
                t2[f'{method}'] = np.nan

                m[f'{method}'] = np.nan
                m2[f'{method}'] = np.nan

            if not is_missing:

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
    logger.info(f'\nMemory results:\n{m_df}')
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
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_resources_args().parse_args())
