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
from remove import get_rank_df
from rank.remove import get_mean_df
from scipy.stats import gmean


def process(args, out_dir, logger):
    begin = time.time()

    # result containers
    time_rows = []
    time2_rows = []
    mem_rows = []

    color, line, label = pp_util.get_plot_dicts()

    for tree_type in args.tree_type_list:
        logger.info(f'\n{tree_type}')

        for dataset in args.dataset_list:
            logger.info(f'\n{dataset}')

            args.dataset = dataset

            # collect results
            r1 = {}
            for random_state in range(1, args.n_repeat + 1):
                logger.info(f'random state: {random_state}')

                r1[random_state] = {}

                exp_dir = os.path.join(args.in_dir,
                                       dataset,
                                       tree_type,
                                       f'random_state_{random_state}')
                res_list = pp_util.get_results(args, exp_dir, logger, progress_bar=False)
                res_list = pp_util.filter_results(res_list, args.skip)

                for method, d in res_list:
                    r = {}
                    r['mem_GB'] = d['max_rss_MB']  # results were run on Linux
                    r['fit_time'] = d['fit_time']
                    r['inf_time'] = d['inf_time']
                    r['total_time'] = r['fit_time'] + r['inf_time']

                    r1[random_state][label[method]] = r

            t = {'dataset': dataset, 'tree_type': tree_type}
            t2 = t.copy()
            m = t.copy()

            # average over random states
            for method in r1[random_state].keys():
                is_missing = False

                try:
                    time_vals = [r1[rs][method]['fit_time'] for rs in r1.keys()]
                    time2_vals = [r1[rs][method]['inf_time'] for rs in r1.keys()]
                    mem_vals = [r1[rs][method]['mem_GB'] for rs in r1.keys()]

                    time_mean = np.mean(time_vals)
                    time2_mean = np.mean(time2_vals)
                    mem_mean = np.mean(mem_vals)

                except:
                    is_missing = True
                    logger.info(f'partially missing: {method}')

                    t[f'{method}'] = np.nan
                    t2[f'{method}'] = np.nan
                    m[f'{method}'] = np.nan

                if not is_missing:
                    t[f'{method}'] = time_mean
                    t2[f'{method}'] = time2_mean
                    m[f'{method}'] = mem_mean

            time_rows.append(t)
            time2_rows.append(t2)
            mem_rows.append(m)

    # compile results
    t_df = pd.DataFrame(time_rows)
    t2_df = pd.DataFrame(time2_rows)
    m_df = pd.DataFrame(mem_rows)

    logger.info(f'\nFit time (s):\n{t_df}')
    logger.info(f'\nInfluence time (s):\n{t2_df}')
    logger.info(f'\nMemory results:\n{m_df}')
    logger.info(f'\nSaving results to {out_dir}...')

    t_df.to_csv(os.path.join(out_dir, 'fit_time.csv'), index=None)
    t2_df.to_csv(os.path.join(out_dir, 'influence_time.csv'), index=None)
    m_df.to_csv(os.path.join(out_dir, 'mem.csv'), index=None)

    logger.info('\ndropping NaN rows...')
    t_df = t_df.dropna()
    t2_df = t2_df.dropna()
    m_df = m_df.dropna()

    # swap fit and influence times for SubSample and LOO
    for c in ['SubSample', 'LOO']:
        t_df[f'{c}2'] = t2_df[c]
        t2_df[c] = t_df[c]
        t_df[c] = t_df[f'{c}2']
        del t_df[f'{c}2']

    # LOO and SubSample correction
    ls_fp = os.path.join(out_dir, 'loo_subsample.csv')
    if os.path.exists(ls_fp):
        ls_df = pd.read_csv(ls_fp)
        ls_df = ls_df[ls_df['tree_type'].isin(t_df['tree_type'].unique())]
        for c in ['SubSample', 'LOO']:
            t_df[c] = t_df[c].values - ls_df[c].values
            t2_df[c] = t2_df[c].values + ls_df[c].values

    # average over tree types
    del t_df['tree_type']
    del t2_df['tree_type']
    del m_df['tree_type']

    t_df = t_df.groupby(['dataset']).agg(gmean).reset_index()   
    t2_df = t2_df.groupby(['dataset']).agg(gmean).reset_index()
    m_df = m_df.groupby(['dataset']).agg(gmean).reset_index()

    # get relevant columns
    cols = [x for x in t_df.columns if x not in ['dataset', 'tree_type', 'Random', 'Target', 'Input Sim.']]
    t_df = t_df[cols].copy()
    t2_df = t2_df[cols].copy()
    m_df = m_df[cols].copy()

    # specify ordering
    order = ['TreeSim', 'BoostIn', 'LeafInfSP', 'TREX', 'SubSample', 'LOO', 'LeafRefit', 'LeafInfluence']
    t_df = t_df[order]
    t2_df = t2_df[order]
    m_df = m_df[order]

    # plot
    pp_util.plot_settings(fontsize=18)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # alternate above and below for labels
    labels = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(order)]

    ax = axs[0]
    t_df.boxplot([c for c in t_df.columns], ax=ax)
    ax.set_ylabel('Fit time (s)')
    ax.set_yscale('log')
    ax.set_xticklabels(labels)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])

    ax = axs[1]
    t2_df.boxplot([c for c in t2_df.columns], ax=ax)
    ax.set_ylabel('Influence time (s)')
    ax.set_yscale('log')
    ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'resources.pdf'), bbox_inches='tight')

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    # create output directory
    out_dir = os.path.join(args.out_dir,
                           'summary',
                           '+'.join(args.tree_type_list))

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_resources_args().parse_args())
