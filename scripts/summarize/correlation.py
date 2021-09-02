"""
Aggregate correlations.
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
from experiments import util


def experiment(args, logger, in_dir, out_dir):
    begin = time.time()

    idx_dict = {'trex': 0, 'similarity': 1, 'boostin2': 2, 'leaf_influenceSP': 3,
                'subsample': 4, 'loo': 5, 'target': 6}

    names = ['trex', ' similarity', 'boostin2', 'leaf_influenceSP', 'subsample', 'loo', 'target']

    n_method = len(names)

    p_mean_list = []
    s_mean_list = []
    j_mean_list = []

    p_std_list = []
    s_std_list = []
    j_std_list = []

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

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, in_dir, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='output/plot/correlation/')
    parser.add_argument('--out_dir', type=str, default='output/plot/correlation/')

    # Data settings
    parser.add_argument('--dataset_list', type=str, nargs='+',
                        default=['adult', 'bank_marketing', 'bean', 'compas',
                                 'concrete', 'credit_card', 'diabetes', 'energy',
                                 'flight_delays', 'german_credit', 'htru2', 'life',
                                 'msd', 'naval', 'no_show', 'obesity', 'power', 'protein',
                                 'spambase', 'surgical', 'twitter', 'vaccine', 'wine'])

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Explainer settings
    parser.add_argument('--leaf_scale', type=float, default=-1.0)  # BoostIn
    parser.add_argument('--local_op', type=str, default='normal')  # BoostIn

    parser.add_argument('--update_set', type=int, default=0)  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default='dot_prod')  # Similarity

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex & Similarity
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--sub_frac', type=float, default=0.7)  # SubSample
    parser.add_argument('--n_iter', type=int, default=4000)  # SubSample

    parser.add_argument('--global_op', type=str, default='self')  # TREX, LOO, and DShap

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random

    # Experiment settings
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
    parser.add_argument('--n_ckpt', type=int, default=50)
    parser.add_argument('--method', type=str, nargs='+',
                        default=['target', 'similarity', 'boostin2',
                                 'trex', 'leaf_influenceSP', 'loo', 'subsample'])  # no minority, loss
    parser.add_argument('--zoom', type=float, nargs='+', default=[1.0])

    args = parser.parse_args()
    main(args)
