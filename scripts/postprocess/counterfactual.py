"""
Characterize the examples being deleted.
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
    results = pp_util.get_results(args, in_dir=args.in_dir, logger=logger)
    results = filter_results(results, args.skip)

    color, line, label = pp_util.get_plot_dicts()

    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    
    for i in range(len(results)):
        method, res = results[i]

        df = res['df']
        n_correct = len(df[df['status'] == 'success']) + len(df[df['status'] == 'fail'])

        df = df[df['status'] == 'success']
        if len(df) == 0:
            continue

        df = df.sort_values('n_edits')
        df['count'] = np.arange(1, len(df) + 1)

        ax.plot(df['frac_edits'] * 100, df['count'], label=label[method], color=color[method],
                linestyle=line[method], alpha=0.75)
        ax.set_xlabel('% train targets poisoned')
        ax.set_ylabel(f'No. correct test preds. flipped (cum.)')

        ax2.plot(df['n_edits'], df['count'], label=label[method], color=color[method],
                 linestyle=line[method], alpha=0.75)
        ax2.set_xlabel('No. train targets poisoned')

    ax.axhline(n_correct, label='No. correct test preds.', color='k', linestyle='--', linewidth=1)
    ax.legend(fontsize=6)

    plt_dir = os.path.join(args.out_dir, args.tree_type)
    suffix = ''

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
    parser.add_argument('--in_dir', type=str, default='temp_counterfactual/')
    parser.add_argument('--out_dir', type=str, default='output/plot/counterfactual/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
    parser.add_argument('--n_ckpt', type=int, default=50)

    # Method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['random', 'target', 'boostin2', 'boostin4', 'trex', 'similarity',
                                 'leaf_influenceSP', 'loo', 'subsample'])
    parser.add_argument('--skip', type=str, nargs='+',
                        default=['minority', 'loss'])

    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal', 'sign', 'sim'])  # BoostIn
    parser.add_argument('--update_set', type=int, nargs='+', default=[-1, 0])  # LeafInfluence

    parser.add_argument('--similarity', type=str, nargs='+', default=['dot_prod'])  # Similarity

    parser.add_argument('--kernel', type=str, nargs='+', default=['lpw'])  # Trex & Similarity
    parser.add_argument('--target', type=str, nargs='+', default=['actual'])  # Trex
    parser.add_argument('--lmbd', type=float, nargs='+', default=[0.003])  # Trex
    parser.add_argument('--n_epoch', type=str, nargs='+', default=[3000])  # Trex

    parser.add_argument('--trunc_frac', type=float, nargs='+', default=[0.25])  # DShap
    parser.add_argument('--check_every', type=int, nargs='+', default=[100])  # DShap

    parser.add_argument('--sub_frac', type=float, nargs='+', default=[0.7])  # SubSample
    parser.add_argument('--n_iter', type=int, nargs='+', default=[4000])  # SubSample

    parser.add_argument('--global_op', type=str, nargs='+', default=['self', 'expected'])  # TREX, LOO, DShap

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random

    args = parser.parse_args()
    main(args)
