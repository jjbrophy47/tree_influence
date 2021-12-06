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
            m = t.copy()

            # average over random states
            for method in r1[random_state].keys():
                is_missing = False

                try:
                    time_vals = [r1[rs][method]['total_time'] for rs in r1.keys()]
                    time_mean = np.mean(time_vals)

                    mem_vals = [r1[rs][method]['mem_GB'] for rs in r1.keys()]
                    mem_mean = np.mean(mem_vals)

                except:
                    is_missing = True
                    logger.info(f'partially missing: {method}')

                    t[f'{method}'] = np.nan
                    m[f'{method}'] = np.nan

                if not is_missing:
                    t[f'{method}'] = time_mean
                    m[f'{method}'] = mem_mean

            time_rows.append(t)
            mem_rows.append(m)

    # compile results
    t_df = pd.DataFrame(time_rows)
    m_df = pd.DataFrame(mem_rows)

    # use time results if already availabled
    fp = os.path.join(out_dir, 'time.csv')
    if os.path.exists(fp) and args.use_existing:
        logger.info('\nUsing saved time.csv...')
        t_df = pd.read_csv(fp)

    logger.info(f'\nTime results:\n{t_df}')
    logger.info(f'\nMemory results:\n{m_df}')
    logger.info(f'\nSaving results to {out_dir}...')

    t_df.to_csv(os.path.join(out_dir, 'time.csv'), index=None)
    m_df.to_csv(os.path.join(out_dir, 'mem.csv'), index=None)

    logger.info('\ndropping NaN rows...')
    t_df = t_df.dropna()
    m_df = m_df.dropna()

    # get avg. rankings
    group_cols = ['dataset']
    skip_cols = ['dataset', 'tree_type', 'remove_frac']
    remove_cols = ['Random', 'Target', 'Input Sim.']

    t_rank_df = get_rank_df(t_df, skip_cols=skip_cols, remove_cols=remove_cols, ascending=True)  # get ranks
    t_avg_rank_df = t_rank_df.groupby(group_cols).mean().reset_index()  # average over tree types
    t_mean_rank_df = get_mean_df(t_avg_rank_df, skip_cols=skip_cols)  # average over datasets

    # average over tree types
    del t_df['tree_type']
    del m_df['tree_type']

    t_df = t_df.groupby(['dataset']).agg(gmean).reset_index()
    m_df = m_df.groupby(['dataset']).agg(gmean).reset_index()

    # get relevant columns
    cols = [x for x in t_df.columns if x not in ['dataset', 'tree_type', 'Random', 'Target', 'Input Sim.']]
    t_df = t_df[cols].copy()
    m_df = m_df[cols].copy()

    # compute relative speedups
    base_method = 'TreeSim'  # fastest method

    t_df.loc[:, cols] = t_df.loc[:, cols].div(t_df[base_method], axis=0)
    m_df.loc[:, cols] = m_df.loc[:, cols].div(m_df[base_method], axis=0)

    logger.info(f'\nAvg. time rankings:\n{t_mean_rank_df}')

    logger.info(f'\nRelative time results:\n{t_df}')
    logger.info(f'\nRelative memory results:\n{m_df}')

    # remove base method
    del t_df[base_method]
    del m_df[base_method]
    cols = [x for x in cols if x != base_method]

    # specify ordering
    order = ['BoostIn', 'LeafInfSP', 'TREX', 'SubSample', 'LOO', 'LeafRefit', 'LeafInfluence']
    t_df = t_df[order]
    m_df = m_df[order]

    # compute geo. mean/s.d. over datasets
    t_mean, t_sd = gmean(t_df[order].values, axis=0), np.std(t_df[order].values, axis=0)
    m_mean, m_sd = gmean(m_df[order].values, axis=0), np.std(m_df[order].values, axis=0)

    # plot
    pp_util.plot_settings(fontsize=18)
    width = 10
    height = pp_util.get_height(width * 0.8)

    fig, ax = plt.subplots(figsize=(width, height))

    # alternate above and below for labels
    labels = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(order)]

    ax.bar(labels, t_mean, color='#ff7600', label='SDS')
    ax.set_ylabel('Slowdown relative to TreeSim')
    ax.set_yscale('log')
    ax.set_ylim(1, None)
    ax.grid(True, which='major', axis='y')
    ax.set_axisbelow(True)
    # ax.legend(framealpha=1.0)

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
