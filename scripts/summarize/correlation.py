"""
Summarize correlations.
"""
import os
import sys
import time
import tqdm
import hashlib
import argparse
import resource
import seaborn as sns
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import sem

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util
from config import summ_args


def experiment(args, logger, in_dir, out_dir):
    begin = time.time()

    p_mean_list = []
    s_mean_list = []
    j_mean_list = []

    p_std_list = []
    s_std_list = []
    j_std_list = []

    idx_dict = None

    # get correlation results
    n_finish = 0

    logger.info('')
    for dataset in args.dataset_list:
        fp = os.path.join(in_dir, dataset, 'results.npy')

        if not os.path.exists(fp):
            logger.info(f'skipping {dataset}...')
            continue

        res = np.load(fp, allow_pickle=True)[()]

        p_mean_list.append(res['p_mean_mat'])
        s_mean_list.append(res['s_mean_mat'])
        j_mean_list.append(res['j_mean_mat'])

        p_std_list.append(res['p_std_mat'])
        s_std_list.append(res['s_std_mat'])
        j_std_list.append(res['j_std_mat'])

        # sanity check
        if idx_dict is None:
            idx_dict = res['idx_dict']

        else:
            assert idx_dict == res['idx_dict']

    inv_idx_dict = {v: k for k, v in idx_dict.items()}
    names = [inv_idx_dict[i] for i in range(len(inv_idx_dict))]
    n_method = len(names)

    p_mean = np.dstack(p_mean_list).mean(axis=2)
    s_mean = np.dstack(s_mean_list).mean(axis=2)
    j_mean = np.dstack(j_mean_list).mean(axis=2)

    p_std = np.dstack(p_std_list).mean(axis=2)
    s_std = np.dstack(s_std_list).mean(axis=2)
    j_std = np.dstack(j_std_list).mean(axis=2)

    p_mean_df = pd.DataFrame(p_mean, columns=names, index=names)
    s_mean_df = pd.DataFrame(s_mean, columns=names, index=names)
    j_mean_df = pd.DataFrame(j_mean, columns=names, index=names)

    logger.info(f'\nPearson results:\n{p_mean_df}')
    logger.info(f'\nSpearman results:\n{s_mean_df}')
    logger.info(f'\nJaccard (10%) results:\n{j_mean_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    p_mean_df.to_csv(os.path.join(out_dir, 'pearson.csv'))
    s_mean_df.to_csv(os.path.join(out_dir, 'spearman.csv'))
    j_mean_df.to_csv(os.path.join(out_dir, 'jaccard_10.csv'))

    # plot correlations
    mask = None

    fig, ax = plt.subplots()
    sns.heatmap(p_mean, xticklabels=names, yticklabels=names, ax=ax,
                cmap='YlGnBu', mask=mask, fmt='.2f', cbar=True, annot=True)
    ax.set_title(f'Pearson')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(out_dir, f'pearson.png'), bbox_inches='tight')

    fig, ax = plt.subplots()
    sns.heatmap(s_mean, xticklabels=names, yticklabels=names, ax=ax,
                cmap='YlGnBu', mask=mask, fmt='.2f', annot=True)
    ax.set_title('Spearman')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(out_dir, f'spearman.png'), bbox_inches='tight')

    fig, ax = plt.subplots()
    sns.heatmap(j_mean, xticklabels=names, yticklabels=names, ax=ax,
                cmap='YlGnBu', mask=mask, fmt='.2f', annot=True)
    ax.set_title('Jaccard (first 10% of sorted)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(out_dir, f'jaccard_10.png'), bbox_inches='tight')

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    # create input dir
    in_dir = os.path.join(args.in_dir,
                          args.tree_type)

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           'summary')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, in_dir, out_dir)


if __name__ == '__main__':
    main(summ_args.get_correlation_args().parse_args())
