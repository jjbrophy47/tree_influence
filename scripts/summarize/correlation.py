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
from postprocess import util as pp_util
from experiments import util as exp_util
from config import summ_args


def reorder_heatmap(arr, idxs):
    """
    Reorder heatmap based on the given indices.

    Input
        arr: 2d Array.
        idxs: 1d array of indices.

    Return
        2d array with relevant positions swapped.
    """
    assert arr.ndim == 2
    assert idxs.ndim == 1 and len(idxs) == len(arr)

    idxs = idxs.copy()

    # make sure idxs starts at 0 and increse by 1
    if np.min(idxs) > 0 or np.max(idxs) >= len(idxs):
        sorted_positions = np.argsort(idxs)
        new_idx_vals = np.arange(len(idxs))

        for pos, new_idx_val in zip(sorted_positions, new_idx_vals):
            idxs[pos] = new_idx_val

    # build pairwise dictionary
    pw = {}
    for i in range(len(arr)):
        for j in range(len(arr)):
            pw[f'{i}_{j}'] = arr[i, j]

    # rebuild heatmap with specified ordering
    res = np.zeros(arr.shape)
    for i1, i2 in zip(np.arange(len(arr)), idxs):
        for j1, j2 in zip(np.arange(len(arr)), idxs):
            res[i1, j1] = pw[f'{i2}_{j2}']

    return res


def experiment(args, logger, out_dir):
    begin = time.time()

    p_mean_list = []
    s_mean_list = []
    j_mean_list = []

    idx_dict = None

    # hardcode classification and regression datasets
    if args.out_sub_dir == 'classification':
        args.dataset_list = ['adult', 'bank_marketing', 'bean', 'compas', 'credit_card',
                             'diabetes', 'flight_delays', 'german_credit', 'htru2', 'no_show',
                             'spambase', 'surgical', 'twitter', 'vaccine']

    elif args.out_sub_dir == 'regression':
        args.dataset_list = ['concrete', 'energy', 'life', 'naval', 'obesity',
                             'power', 'protein', 'wine']

    # get correlation results
    n_finish = 0

    for tree_type in args.tree_type_list:
        logger.info(f'\n{tree_type}')

        for dataset in args.dataset_list:
            logger.info(f'{dataset}')

            res_dir = os.path.join(args.in_dir, tree_type, dataset)
            if args.in_sub_dir is not None:
                res_dir = os.path.join(res_dir, args.in_sub_dir)

            fp = os.path.join(res_dir, 'results.npy')

            if not os.path.exists(fp):
                logger.info(f'skipping {fp}...')
                continue

            res = np.load(fp, allow_pickle=True)[()]

            p_mean_list.append(res['p_mean_mat'])
            s_mean_list.append(res['s_mean_mat'])
            j_mean_list.append(res['j_mean_mat'])

            # sanity check
            if idx_dict is None:
                idx_dict = res['idx_dict']

            else:
                assert idx_dict == res['idx_dict']

    label_dict = {'target': 'RandomSL', 'leaf_sim': 'TreeSim', 'boostin': 'BoostIn',
                  'trex': 'TREX', 'leaf_infSP': 'LeafInfSP', 'loo': 'LOO',
                  'subsample': 'SubSample', 'leaf_inf': 'LeafInfluence', 'leaf_refit': 'LeafRefit'}

    inv_idx_dict = {v: k for k, v in idx_dict.items()}
    idxs = np.array([k for k, v in inv_idx_dict.items() if v in args.method_list], dtype=np.int32)
    names = [inv_idx_dict[i] for i in idxs]
    n_method = len(names)

    p_mean = np.dstack(p_mean_list).mean(axis=2)
    s_mean = np.dstack(s_mean_list).mean(axis=2)
    j_mean = np.dstack(j_mean_list).mean(axis=2)

    p_mean = p_mean[np.ix_(idxs, idxs)]
    s_mean = s_mean[np.ix_(idxs, idxs)]
    j_mean = j_mean[np.ix_(idxs, idxs)]

    # reorder heatmap
    order = ['boostin', 'leaf_infSP', 'trex', 'leaf_sim', 'subsample', 'loo']

    if args.out_sub_dir == 'li':
        order = ['boostin', 'leaf_infSP', 'leaf_sim', 'trex', 'subsample', 'loo', 'leaf_refit', 'leaf_inf']

    new_idxs = np.array([idx_dict[m] for m in order if m in args.method_list], dtype=np.int32)

    p_mean = reorder_heatmap(p_mean, new_idxs)
    s_mean = reorder_heatmap(s_mean, new_idxs)
    j_mean = reorder_heatmap(j_mean, new_idxs)

    p_mean_df = pd.DataFrame(p_mean, columns=order, index=order)
    s_mean_df = pd.DataFrame(s_mean, columns=order, index=order)
    j_mean_df = pd.DataFrame(j_mean, columns=order, index=order)

    logger.info(f'\nPearson results:\n{p_mean_df}')
    logger.info(f'\nSpearman results:\n{s_mean_df}')
    logger.info(f'\nJaccard (10%) results:\n{j_mean_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    p_mean_df.to_csv(os.path.join(out_dir, 'pearson.csv'))
    s_mean_df.to_csv(os.path.join(out_dir, 'spearman.csv'))
    j_mean_df.to_csv(os.path.join(out_dir, 'jaccard_10.csv'))

    # plot correlations
    cmap = 'Oranges' if args.out_sub_dir == 'li' else 'Blues'
    fontsize = 15 if args.out_sub_dir == 'li' else 15

    pp_util.plot_settings(fontsize=fontsize)

    # mask = None
    mask = np.triu(np.ones_like(p_mean, dtype=bool))  # uncomment for mask

    labels = [label_dict[name] for name in order]
    labels_x = [c if i % 2 != 0 else f'\n{c}' for i, c in enumerate(labels)]

    suffix = '_li' if args.out_sub_dir == 'li' else ''

    fig, ax = plt.subplots()
    sns.heatmap(p_mean, xticklabels=labels, yticklabels=labels, ax=ax,
                cmap='Greens', mask=mask, fmt='.2f', cbar=True, annot=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(out_dir, f'pearson{suffix}.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots()
    sns.heatmap(s_mean, xticklabels=labels, yticklabels=labels, ax=ax,
                cmap=cmap, mask=mask, fmt='.2f', annot=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(out_dir, f'spearman{suffix}.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots()
    sns.heatmap(j_mean, xticklabels=labels, yticklabels=labels, ax=ax,
                cmap='Blues', mask=mask, fmt='.2f', annot=False)
    ax.set_title('Jaccard (first 10% of sorted)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(out_dir, f'jaccard_10{suffix}.pdf'), bbox_inches='tight')

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    # create output dir
    assert len(args.tree_type_list) > 0
    out_dir = os.path.join(args.out_dir,
                           'summary',
                           '+'.join(args.tree_type_list))

    if args.out_sub_dir is not None:
        out_dir = os.path.join(out_dir, args.out_sub_dir)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, out_dir)


if __name__ == '__main__':
    main(summ_args.get_correlation_args().parse_args())
