"""
Measure correlation between two influence methods.
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


def jaccard_similarity(inf1, inf2):
    """
    Return |inf1 intersect inf2| / |inf1 union inf2|.
    """
    s1 = set(inf1)
    s2 = set(inf2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def get_correlation(inf1, inf2, jaccard_frac=0.1):
    """
    Compute avg. correlation between the given influences.
    """
    inf1 = np.where(inf1 == np.inf, 1000, inf1)  # replace inf with a large constant
    inf2 = np.where(inf2 == np.inf, 1000, inf2)

    inf1 = np.nan_to_num(inf1)  # replace nan with a small number
    inf2 = np.nan_to_num(inf2)

    pearson = np.zeros(inf1.shape[1], dtype=np.float32)
    spearman = np.zeros(inf1.shape[1], dtype=np.float32)
    jaccard = np.zeros(inf1.shape[1], dtype=np.float32)

    for i in range(inf1.shape[1]):
        inf1i = inf1[:, i]
        inf2i = inf2[:, i]

        # check if either array is constant
        if np.all(inf1i == inf1i[0]) or np.all(inf2i == inf2i[0]):
            print('Constant detected!')
            continue

        pearson[i] = pearsonr(inf1i, inf2i)[0]
        spearman[i] = spearmanr(inf1i, inf2i)[0]

        n_cutoff = int(len(inf1i) * jaccard_frac)
        jaccard[i] = jaccard_similarity(np.argsort(inf1i)[-n_cutoff:], np.argsort(inf2i)[-n_cutoff:])

    mean_p = np.mean(pearson)
    mean_s = np.mean(spearman)
    mean_j = np.mean(jaccard)

    std_p = np.std(pearson)
    std_s = np.std(spearman)
    std_j = np.std(jaccard)

    return mean_p, std_p, mean_s, std_s, mean_j, std_j


def experiment(args, logger, in_dir_list, out_dir):

    # initialize experiment
    begin = time.time()

    n_method = len(in_dir_list)

    p_mean_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    s_mean_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    j_mean_mat = np.full((n_method, n_method), 1, dtype=np.float32)

    p_std_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    s_std_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    j_std_mat = np.full((n_method, n_method), 1, dtype=np.float32)

    logger.info(f'\nno. methods: {n_method:,}, no. comparisons: {n_method ** 2}')

    # get influence results
    names = []
    n_finish = 0

    for i, (method1, in_dir1) in enumerate(in_dir_list):
        inf_res1 = np.load(os.path.join(in_dir1, 'results.npy'), allow_pickle=True)[()]
        names.append(method1)

        for j, (method2, in_dir2) in enumerate(in_dir_list):

            if method1 == method2:
                n_finish += 1
                logger.info(f'no. finish: {n_finish:>10,} / {n_method ** 2}, cum. time: {time.time() - begin:.3f}s')
                continue

            inf_res2 = np.load(os.path.join(in_dir2, 'results.npy'), allow_pickle=True)[()]
            mean_p, std_p, mean_s, std_s, mean_j, std_j = get_correlation(inf_res1['influence'], inf_res2['influence'])

            p_mean_mat[i, j] = mean_p
            s_mean_mat[i, j] = mean_s
            j_mean_mat[i, j] = mean_j

            p_std_mat[i, j] = std_p
            s_std_mat[i, j] = std_s
            j_std_mat[i, j] = std_j

            n_finish += 1
            logger.info(f'no. finish: {n_finish:>10,} / {n_method ** 2}, cum. time: {time.time() - begin:.3f}s')

            n_test = inf_res1['influence'].shape[1]

    logger.info(f'\ntotal time: {time.time() - begin:.3f}s')

    # plot correlations
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # mask = np.zeros_like(p_mean_mat)
    # mask[np.triu_indices_from(mask)] = True

    mask = None

    ax = axs[0]
    sns.heatmap(p_mean_mat, xticklabels=names, yticklabels=names, ax=ax,
                cmap='YlGnBu', mask=mask, fmt='.2f', cbar=True, annot=True)
    ax.set_title(f'Pearson (Avg. over {n_test} test)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[1]
    sns.heatmap(s_mean_mat, xticklabels=names, yticklabels=names, ax=ax,
                cmap='YlGnBu', mask=mask, fmt='.2f', annot=True)
    ax.set_title('Spearman')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax = axs[2]
    sns.heatmap(j_mean_mat, xticklabels=names, yticklabels=names, ax=ax,
                cmap='YlGnBu', mask=mask, fmt='.2f', annot=True)
    ax.set_title('Jaccard (first 10% of sorted)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')
    plt.show()


def main(args):

    # experiment hash_str
    exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
                'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = util.dict_to_hash(exp_dict)

    in_dir_list = []
    for method in args.method:
        _, method_hash = util.explainer_params_to_dict(method, vars(args))
        in_dir = os.path.join(args.in_dir,
                              args.dataset,
                              args.tree_type,
                              f'exp_{exp_hash}',
                              f'{method}_{method_hash}')
        in_dir_list.append((method, in_dir))

    # create output dir
    out_dir = os.path.join(args.out_dir, args.tree_type)
    log_dir = os.path.join(args.out_dir, args.tree_type, 'logs')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, in_dir_list, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_influence')
    parser.add_argument('--out_dir', type=str, default='output/plot/correlation/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='synthetic_regression')

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
