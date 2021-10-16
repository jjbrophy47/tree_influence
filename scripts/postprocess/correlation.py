"""
Measure correlation between influence methods.
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
from experiments import util as exp_util
from config import post_args


def jaccard_similarity(inf1, inf2):
    """
    Return |inf1 intersect inf2| / |inf1 union inf2|.
    """
    s1 = set(inf1)
    s2 = set(inf2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def get_correlation(inf1, inf2, jaccard_frac=0.1, logger=None):
    """
    Compute avg. correlation between the given influences.
    """
    inf1 = np.where(inf1 == np.inf, 1000, inf1)  # replace inf with a large constant
    inf2 = np.where(inf2 == np.inf, 1000, inf2)

    inf1 = np.where(inf1 > 1e300, 1000, inf1)  # replace very large numbers with a large constant
    inf2 = np.where(inf2 > 1e300, 1000, inf2)

    inf1 = np.where(inf1 < -1e300, -1000, inf1)  # replace very small numbers with a large constants
    inf2 = np.where(inf2 < -1e300, -1000, inf2)

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
            if logger:
                logger.info(f'Constant detected! Test no. {i}')
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
    begin = time.time()

    idx_dict = {method: i for i, method in enumerate(args.method_list)}
    names = args.method_list

    n_method = len(idx_dict)

    p_mean_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    s_mean_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    j_mean_mat = np.full((n_method, n_method), 1, dtype=np.float32)

    p_std_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    s_std_mat = np.full((n_method, n_method), 1, dtype=np.float32)
    j_std_mat = np.full((n_method, n_method), 1, dtype=np.float32)

    logger.info(f'\nno. methods: {n_method:,}, no. comparisons: {n_method ** 2}')

    # get influence results
    n_finish = 0

    for i, (method1, in_dir1) in enumerate(in_dir_list):

        try:
            inf_res1 = np.load(os.path.join(in_dir1, 'results.npy'), allow_pickle=True)[()]
        except:
            inf_res1 = None

        for j, (method2, in_dir2) in enumerate(in_dir_list):

            if method1 == method2:
                n_finish += 1
                logger.info(f'no. finish: {n_finish:>10,} / {n_method ** 2}, cum. time: {time.time() - begin:.3f}s')
                continue

            try:
                inf_res2 = np.load(os.path.join(in_dir2, 'results.npy'), allow_pickle=True)[()]
                mean_p, std_p, mean_s, std_s, mean_j, std_j = get_correlation(inf_res1['influence'],
                                                                              inf_res2['influence'], logger=logger)

            except:
                logger.info(f'failed to read {method2} or {method1}')
                mean_p, std_p, mean_s, std_s, mean_j, std_j = 0, 0, 0, 0, 0, 0

            if inf_res1 is None:
                mean_p, std_p, mean_s, std_s, mean_j, std_j = 0, 0, 0, 0, 0, 0

            idx1 = idx_dict[method1]
            idx2 = idx_dict[method2]

            p_mean_mat[idx1, idx2] = mean_p
            s_mean_mat[idx1, idx2] = mean_s
            j_mean_mat[idx1, idx2] = mean_j

            p_std_mat[idx1, idx2] = std_p
            s_std_mat[idx1, idx2] = std_s
            j_std_mat[idx1, idx2] = std_j

            n_finish += 1
            logger.info(f'no. finish: {n_finish:>10,} / {n_method ** 2}, cum. time: {time.time() - begin:.3f}s')

            if inf_res1 is not None:
                n_test = inf_res1['influence'].shape[1]

    logger.info(f'\ntotal time: {time.time() - begin:.3f}s')

    # plot correlations
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

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

    logger.info(f'\nSaving results to {out_dir}...')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'plot.png'), bbox_inches='tight')

    result = {}
    result['p_mean_mat'] = p_mean_mat
    result['s_mean_mat'] = s_mean_mat
    result['j_mean_mat'] = j_mean_mat
    result['p_std_mat'] = p_std_mat
    result['s_std_mat'] = s_std_mat
    result['j_std_mat'] = j_std_mat
    result['idx_dict'] = idx_dict

    logger.info(f'\nResults:\n{result}')
    np.save(os.path.join(out_dir, 'results.npy'), result)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    # experiment hash string
    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    in_dir_list = []
    for method in args.method_list:

        if method in args.skip:
            continue

        _, method_hash = exp_util.explainer_params_to_dict(method, vars(args))

        in_dir = os.path.join(args.in_dir,
                              args.dataset,
                              args.tree_type,
                              f'exp_{exp_hash}',
                              f'{method}_{method_hash}')
        in_dir_list.append((method, in_dir))

    # create output dir
    out_dir = os.path.join(args.out_dir, args.tree_type, args.dataset)

    if args.custom_dir is not None:
        out_dir = os.path.join(out_dir, args.custom_dir)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, in_dir_list, out_dir)


if __name__ == '__main__':
    main(post_args.get_correlation_args().parse_args())
