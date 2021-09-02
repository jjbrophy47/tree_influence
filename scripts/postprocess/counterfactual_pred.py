"""
Predict the counterfactual influence set size.
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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util as pp_util
from experiments import util
from leaf_analysis import filter_results


def predict_influence_set(influence):
    """
    Use the influence values to predict the no. examples,
    that when flipped, flip the prediction of the test example.

    Returns 1d array of shape=(influence.shape[1],).
    """
    result = np.zeros(influence.shape[1], dtype=np.int32)

    for test_idx in range(influence.shape[1]):
        inf = influence[:, test_idx]

        # print(inf[np.argsort(inf)[::-1]])
        # print(np.sum(inf))

        sum_total = np.sum(inf)
        for i, idx in enumerate(np.argsort(inf)[::-1]):
            sum_total -= inf[idx] * 2

            if sum_total < 0:
                result[test_idx] = i + 1
                break

    return result


def experiment(args, logger, out_dir):

    # initialize experiment
    begin = time.time()

    # get dataset
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # get results
    inf_res = filter_results(pp_util.get_results(args, in_dir=args.in_dir1, logger=logger), args.skip)
    cf_res = filter_results(pp_util.get_results(args, in_dir=args.in_dir2, logger=logger), args.skip)

    color, line, label = pp_util.get_plot_dicts()

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()
    
    for i, (method1, cfr) in enumerate(cf_res):
        for method2, ifr in inf_res:
            if method2 == method1:

                df = cfr['df']
                qf = df[df['status'] == 'success']
                if len(qf) == 0:
                    continue

                idxs = qf.index.values
                n_edits = df['n_edits'].values[idxs]
                pred_n_edits = predict_influence_set(ifr['influence'])[idxs]

                p_corr = pearsonr(n_edits, pred_n_edits)[0]
                s_corr = spearmanr(n_edits, pred_n_edits)[0]
                mse = mean_squared_error(n_edits, pred_n_edits)

                ax = axs[i]
                ax.scatter(n_edits, pred_n_edits, label=f'P: {p_corr:.3f}, S: {s_corr:.3f}, MSE: {mse:.3f}')
                ax.plot(n_edits, n_edits, linestyle='-', color='k', label=f'Ideal (n={len(qf):,})')
                ax.set_xlabel('No. edits')
                ax.set_ylabel('Predicted no. edits')
                ax.set_title(f'{label[method1]}')
                ax.legend(fontsize=7)

    plt_dir = os.path.join(args.out_dir, args.tree_type)

    os.makedirs(plt_dir, exist_ok=True)
    fp = os.path.join(plt_dir, f'{args.dataset}')

    plt.tight_layout()
    plt.savefig(fp + '.png', bbox_inches='tight')

    logger.info(f'\nsaving results to {fp}...')


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
    parser.add_argument('--in_dir1', type=str, default='temp_influence/')
    parser.add_argument('--in_dir2', type=str, default='temp_counterfactual/')
    parser.add_argument('--out_dir', type=str, default='output/plot/counterfactual_pred/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
    parser.add_argument('--n_ckpt', type=int, default=50)

    # Method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['trex', 'similarity', 'boostin2',
                                 'leaf_influenceSP', 'subsample', 'loo'])
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
