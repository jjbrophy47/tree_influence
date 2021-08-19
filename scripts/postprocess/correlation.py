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
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import sem

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def experiment(args, logger, in_dir1, in_dir2, out_dir):

    # initialize experiment
    begin = time.time()

    # get influence results
    inf_res1 = np.load(os.path.join(in_dir1, 'results.npy'), allow_pickle=True)[()]
    inf_res2 = np.load(os.path.join(in_dir2, 'results.npy'), allow_pickle=True)[()]

    # evaluate influence ranking
    start = time.time()
    result = {}

    inf1 = inf_res1['influence']
    inf2 = inf_res2['influence']

    # rnk1 = inf_res1['ranking'][:, 0]
    # rnk2 = inf_res2['ranking'][:, 0]

    # print(rnk1, rnk1.shape)
    # print(rnk2, rnk2.shape)

    # n_agree = len(np.where(rnk1 == rnk2)[0])
    # print(n_agree)

    pearson = np.zeros(inf1.shape[1], dtype=np.float32)
    spearman = np.zeros(inf1.shape[1], dtype=np.float32)
    r2 = np.zeros(inf1.shape[1], dtype=np.float32)

    means = np.zeros(2, dtype=np.float32)
    sems = np.zeros(2, dtype=np.float32)

    # average influence values over all test examples
    assert args.inf_obj == 'local'

    # sort influence values based on method1
    for i in tqdm.tqdm(range(inf1.shape[1])):
        idxs = np.argsort(inf1[:, i])[::-1]

        i1 = inf1[:, i][idxs]
        i2 = inf2[:, i][idxs]

        pearson[i] = pearsonr(i1, i2)[0]
        spearman[i] = spearmanr(i1, i2)[0]
        r2[i] = r2_score(i1, i2)

    means[0] = np.mean(pearson, axis=0)
    means[1] = np.mean(spearman, axis=0)
    # means[2] = np.mean(r2, axis=0)

    sems[0] = sem(pearson, axis=0)
    sems[1] = sem(spearman, axis=0)
    # sems[2] = sem(r2, axis=0)

    names = ['Pearson', 'Spearman']

    logger.info(f'\nPearson: {means[0]:.5f} +/- {sems[0]:.5f}')
    logger.info(f'Spearman: {means[1]:.5f} +/- {sems[1]:.5f}')

    # plot correlations
    # fig, ax = plt.subplots()

    # sns.barplot(names, means, yerr=sems)
    # ax.set_title(f'No. test: {pearson.shape[0]}')
    # ax.set_ylabel('Avg. correlation')

    # show_values_on_bars(ax)

    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f'{args.method1}_{args.method2}.png'), bbox_inches='tight')
    # plt.show()


def main(args):

    # get method params and unique settings hash
    _, hash_str1 = util.explainer_params_to_dict(args.method1, vars(args))
    _, hash_str2 = util.explainer_params_to_dict(args.method2, vars(args))

    # experiment hash_str
    exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
                'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = util.dict_to_hash(exp_dict)

    # method1 dir
    in_dir1 = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           f'{args.method1}_{hash_str1}')

    in_dir2 = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           f'{args.method2}_{hash_str2}')

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.inf_obj,
                           args.dataset)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = util.get_logger(os.path.join(out_dir, f'{args.method1}_{args.method2}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, in_dir1, in_dir2, out_dir)


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
    parser.add_argument('--method1', type=str, default='random')
    parser.add_argument('--method2', type=str, default='boostin')
    parser.add_argument('--zoom', type=float, nargs='+', default=[1.0])

    args = parser.parse_args()
    main(args)
