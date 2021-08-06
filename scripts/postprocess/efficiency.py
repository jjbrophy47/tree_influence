"""
Aggregate computational and memory efficiency.
"""
import os
import sys
import time
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.metrics import log_loss

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util as pp_util
from experiments import util
from leaf_analysis import filter_results


def experiment(args, logger, out_dir):

    # initialize experiment
    begin = time.time()

    # get dataset
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # get results
    results = pp_util.get_results(args, args.in_dir, logger)
    results = filter_results(results, args.skip)
    color, line, label = pp_util.get_plot_dicts()

    assert args.inf_obj == 'local'

    method_list = []
    color_list = []
    mem_list = []
    elapsed_list = []

    n_test = None
    
    for i, (method, res) in enumerate(results):

        if i == 0:
            n_test = res['loss'].shape[0]
        else:
            temp = res['loss'].shape[0]
            assert n_test == temp, f'Inconsistent no. test: {temp:,} != {n_test:,}'

        cum_time = res['fit_time'] + res['inf_time']

        # used 3 cpus in parallel for influence
        if 'loo' in method:
            cum_time *= 3

        method_list.append(label[method])
        color_list.append(color[method])
        mem_list.append(res['max_rss_MB'])  # GB if influence was run on linux (Talapas)
        elapsed_list.append(cum_time / n_test)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    pal = sns.color_palette(color_list, len(color_list))

    sns.barplot(x=method_list, y=elapsed_list, palette=pal, ax=axs[0])
    sns.barplot(x=method_list, y=mem_list, palette=pal, ax=axs[1])

    axs[0].set_ylabel('Avg. time per test example (sec.)')
    axs[0].set_yscale('log')
    axs[0].set_title('Computation')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')

    axs[1].set_title('Memory')
    axs[1].set_ylabel('Memory usage (GB)')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')


    plt_dir = os.path.join(args.out_dir, args.inf_obj)
    suffix = f'_{n_test}'

    os.makedirs(plt_dir, exist_ok=True)
    fp = os.path.join(plt_dir, f'{args.dataset}')

    plt.tight_layout()
    plt.savefig(fp + suffix + '.png', bbox_inches='tight')
    plt.show()


def main(args):

    # get method params and unique settings hash
    _, hash_str = util.explainer_params_to_dict(args.method, vars(args))

    # create output dir
    out_dir = os.path.join(args.out_dir)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_influence/')
    parser.add_argument('--out_dir', type=str, default='output/plot/efficiency/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='surgical')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['random', 'target', 'boostin', 'trex', 'similarity',
                                 'leaf_influence', 'loo', 'dshap'])
    parser.add_argument('--skip', type=str, nargs='+',
                        default=['random', 'target', 'minority', 'loss',
                                 'boostin_9e', 'boostin_08', 'boostin_e8', 'boostin_c4'])
    parser.add_argument('--use_leaf', type=int, nargs='+', default=[1, 0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal', 'sign', 'sim'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1, 0])  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default=['dot_prod'])  # Similarity

    parser.add_argument('--kernel', type=str, nargs='+', default=['lpw'])  # Trex & Similarity
    parser.add_argument('--target', type=str, nargs='+', default=['actual'])  # Trex
    parser.add_argument('--lmbd', type=float, nargs='+', default=[0.003])  # Trex
    parser.add_argument('--n_epoch', type=str, nargs='+', default=[3000])  # Trex

    parser.add_argument('--trunc_frac', type=float, nargs='+', default=[0.25])  # DShap
    parser.add_argument('--check_every', type=int, nargs='+', default=[100])  # DShap

    parser.add_argument('--global_op', type=str, nargs='+', default=['self', 'expected'])  # TREX, LOO, DShap

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random

    # Experiment settings
    parser.add_argument('--inf_obj', type=str, default='local')

    args = parser.parse_args()
    main(args)
