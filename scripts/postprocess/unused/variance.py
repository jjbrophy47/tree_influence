"""
Measure influence values vs. confidence/variance and/or error.
"""
import os
import sys
import time
import tqdm
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import sem
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import normaltest

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import exp_util


def experiment(args, logger, in_dir, out_dir):

    # initialize experiment
    begin = time.time()

    # get dataset
    X_train, X_test, y_train, y_test, objective = exp_util.get_data(args.data_dir, args.dataset)

    # get influence results
    inf_res = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]

    # evaluate influence ranking
    start = time.time()
    result = {}

    influence = inf_res['influence']
    y_test_pred = inf_res['y_test_pred']
    test_idxs = inf_res['test_idxs']

    plot_dir = os.path.join(out_dir, args.method)
    os.makedirs(plot_dir, exist_ok=True)

    if objective == 'binary':

        skew_list = []
        kurt_list = []
        sk_list = []
        normal_list = []
        conf_list = []
        net_list = []

        correct_list = []
        incorrect_list = []

        correct0_list = []
        correct1_list = []

        incorrect0_list = []
        incorrect1_list = []

        for idx, test_idx in enumerate(test_idxs):
            proba = y_test_pred[test_idx][0]
            target = y_test[test_idx]

            pred = 0 if proba < 0.5 else 1
            conf = 1 - proba if pred == 0 else proba

            if pred == target:
                correct_list.append(idx)

                if pred == 0:
                    correct0_list.append(idx)

                else:
                    correct1_list.append(idx)

            else:
                incorrect_list.append(idx)

                if pred == 0:
                    incorrect0_list.append(idx)

                else:
                    incorrect1_list.append(idx)

            inf = influence[:, idx]

            skw = skew(inf)
            kurt = kurtosis(inf)
            sk, normal = normaltest(inf)

            pos_inf_idxs = np.where(inf > 0)[0]
            neg_inf_idxs = np.where(inf < 0)[0]

            pos_cum = np.cumsum(inf[pos_inf_idxs][np.argsort(inf[pos_inf_idxs])])
            neg_cum = np.cumsum(np.abs(inf[neg_inf_idxs][np.argsort(np.abs(inf[neg_inf_idxs]))]))

            pos_inf_sum = inf[pos_inf_idxs].sum()
            neg_inf_sum = inf[neg_inf_idxs].sum()
            net_sum = pos_inf_sum + neg_inf_sum

            print(f'[no. {idx:>5} / {len(test_idxs):>10}] {test_idx:>5}: '
                  f'p. sum: {pos_inf_sum:.5f}, n. sum: {neg_inf_sum:.5f}, net sum: {net_sum:.5f}, '
                  f'skew: {skw:.3f}, kurtosis: {kurt:.3f}, sk: {sk:.3f}, '
                  f'pred.: {pred} ({conf * 100:.1f}%), target: {target}')

            skew_list.append(skw)
            kurt_list.append(kurt)
            sk_list.append(sk)
            normal_list.append(normal)
            conf_list.append(conf)

            net_list.append(net_sum)

            # fig, ax = plt.subplots()

            # sns.histplot(inf[pos_inf_idxs], color='green', ax=ax, log_scale=True, cumulative=True, stat='density')
            # sns.histplot(np.abs(inf[neg_inf_idxs]), color='red', ax=ax, log_scale=True, cumulative=True, stat='density')
            # ax.set_title(f'Test {test_idx}, pred.: {pred} ({conf * 100:.1f}%) target: {target}')

            # plt.tight_layout()
            # plt.savefig(os.path.join(plot_dir, f'test_{test_idx}.png'), bbox_inches='tight')

        correct_arr = np.array(correct_list)
        incorrect_arr = np.array(incorrect_list)

        correct0_arr = np.array(correct0_list)
        correct1_arr = np.array(correct1_list)

        incorrect0_arr = np.array(incorrect0_list)
        incorrect1_arr = np.array(incorrect1_list)

        net_arr = np.array(net_list)
        conf_arr = np.array(conf_list)

        net_pearson = pearsonr(net_arr[correct_arr], conf_arr[correct_arr])[0]
        net_spearman = spearmanr(net_arr[correct_arr], conf_arr[correct_arr])[0]

        in_net_pearson = pearsonr(net_arr[incorrect_arr], conf_arr[incorrect_arr])[0]
        in_net_spearman = spearmanr(net_arr[incorrect_arr], conf_arr[incorrect_arr])[0]

        fig, ax = plt.subplots()

        if len(correct0_arr) > 1:
            net0_pearson = pearsonr(net_arr[correct0_arr], conf_arr[correct0_arr])[0]
            net0_spearman = spearmanr(net_arr[correct0_arr], conf_arr[correct0_arr])[0]

            ax.scatter(net_arr[correct0_arr], conf_arr[correct0_arr], color='none', marker='o', edgecolor='green',
                       label=f'correct0 preds.: {net0_pearson:.3f} (P), {net0_spearman:.3f} (S)')

        if len(correct1_arr) > 1:
            net1_pearson = pearsonr(net_arr[correct1_arr], conf_arr[correct1_arr])[0]
            net1_spearman = spearmanr(net_arr[correct1_arr], conf_arr[correct1_arr])[0]

            ax.scatter(net_arr[correct1_arr], conf_arr[correct1_arr], color='green', marker='+',
                       label=f'correct1 preds.: {net1_pearson:.3f} (P), {net1_spearman:.3f} (S)')

        if len(incorrect0_arr) > 1:
            in_net0_pearson = pearsonr(net_arr[incorrect0_arr], conf_arr[incorrect0_arr])[0]
            in_net0_spearman = spearmanr(net_arr[incorrect0_arr], conf_arr[incorrect0_arr])[0]

            ax.scatter(net_arr[incorrect0_arr], conf_arr[incorrect0_arr], color='red', marker='+',
                       label=f'incorrect0 preds.: {in_net0_pearson:.3f} (P), {in_net0_spearman:.3f} (S)')

        if len(incorrect1_arr) > 1:
            in_net1_pearson = pearsonr(net_arr[incorrect1_arr], conf_arr[incorrect1_arr])[0]
            in_net1_spearman = spearmanr(net_arr[incorrect1_arr], conf_arr[incorrect1_arr])[0]

            ax.scatter(net_arr[incorrect1_arr], conf_arr[incorrect1_arr], color='none', marker='o', edgecolor='red',
                       label=f'incorrect1 preds.: {in_net1_pearson:.3f} (P), {in_net1_spearman:.3f} (S)')

        ax.set_xlabel('Sum of influence values')
        ax.set_ylabel('Confidence')

        ax.legend(fontsize=9)

        correlation_dir = os.path.join(out_dir, '..', 'correlation', objective, args.dataset)
        os.makedirs(correlation_dir, exist_ok=True)

        plt.tight_layout()
        plt.savefig(os.path.join(correlation_dir, f'{args.method}.png'), bbox_inches='tight')
        plt.show()

    # regression analysis
    elif objective == 'regression':

        # get unique hash for this experiment setting
        exp_dict = {'n_test': args.n_test}
        exp_hash = exp_util.dict_to_hash(exp_dict)

        # method1 dir
        in_dir2 = os.path.join(args.in_dir2,
                               args.dataset,
                               args.tree_type,
                               f'exp_{exp_hash}',
                               args.var_method)

        # get variance results
        var_res = np.load(os.path.join(in_dir2, 'results.npy'), allow_pickle=True)[()]

        test_idxs2 = var_res['test_idxs']

        assert np.all(test_idxs == test_idxs2)

        targets = y_test[test_idxs]
        pred_mean = var_res['y_test_pred_mean'][test_idxs]
        pred_var = var_res['y_test_pred_scale'][test_idxs]

        inf_sum_arr = np.zeros(len(test_idxs), dtype=np.float32)
        conf_arr = np.zeros(len(test_idxs), dtype=np.float32)

        for idx, test_idx in enumerate(test_idxs):
            inf = influence[:, idx]

            inf_sum_arr[idx] = inf.sum()
            conf_arr[idx] = pred_var[idx]

            pos_inf_idxs = np.where(inf > 0)[0]
            neg_inf_idxs = np.where(inf < 0)[0]

            pos_cum = np.cumsum(inf[pos_inf_idxs][np.argsort(inf[pos_inf_idxs])])
            neg_cum = np.cumsum(np.abs(inf[neg_inf_idxs][np.argsort(np.abs(inf[neg_inf_idxs]))]))

            pos_inf_sum = inf[pos_inf_idxs].sum()
            neg_inf_sum = inf[neg_inf_idxs].sum()
            net_sum = pos_inf_sum + neg_inf_sum

            skw = skew(inf)
            kurt = kurtosis(inf)
            sk, normal = normaltest(inf)

            print(f'[no. {idx:>5} / {len(test_idxs):>10}] {test_idx:>5}: '
                  f'p. sum: {pos_inf_sum:.5f}, n. sum: {neg_inf_sum:.5f}, net sum: {net_sum:.5f}, '
                  f'skew: {skw:.3f}, kurtosis: {kurt:.3f}, sk: {sk:.3f}, '
                  f'pred.: {pred_mean[idx]:.3f} ({pred_var[idx]:.3f}), target: {targets[idx]:.3f}')

            # fig, ax = plt.subplots()
            # ax.plot(np.arange(len(pos_cum)), pos_cum, color='green', label='pos. inf.')
            # ax.plot(np.arange(len(neg_cum)), neg_cum, color='red', label='abs. neg. inf.')
            # ax.set_xlabel('Influence index (sorted from least to greatest)')
            # ax.set_ylabel('Cumulative sum')
            # ax.legend()

            # ax.set_title(f'Test {test_idx}, pred.: {pred_mean[idx]:.3f} ({pred_var[idx]:.3f}) '
            #              f'target: {targets[idx]:.3f}, X_test: {X_test[test_idx]}')

            # plt.tight_layout()
            # plt.savefig(os.path.join(plot_dir, f'test_{test_idx}.png'), bbox_inches='tight')

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        # target vs. predicted
        ax = axs[0][0]
        ax.errorbar(targets, y_test_pred[test_idxs], fmt='.', color='k')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{args.dataset.capitalize()} (1 model; 1.0 frac.)')

        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # influence sum vs. abs. error
        ax = axs[0][1]
        error = targets - y_test_pred[test_idxs][:, 0]
        abs_error = np.abs(error)

        p1 = pearsonr(inf_sum_arr, abs_error)[0]
        s1 = spearmanr(inf_sum_arr, abs_error)[0]

        ax.scatter(inf_sum_arr, abs_error, color='none', marker='o', edgecolor='black',
                   label=f'{p1:.3f} (P), {s1:.3f} (S)')
        ax.set_xlabel('Sum of influence values')
        ax.set_ylabel('L1 error (absolute value)')
        ax.set_title('Influence vs. Absolute Error')
        ax.legend()

        # influence sum vs. error
        ax = axs[0][2]

        p1 = pearsonr(inf_sum_arr, error)[0]
        s1 = spearmanr(inf_sum_arr, error)[0]

        ax.scatter(inf_sum_arr, error, color='none', marker='o', edgecolor='black',
                   label=f'{p1:.3f} (P), {s1:.3f} (S)')
        ax.set_xlabel('Sum of influence values')
        ax.set_ylabel('L1 error')
        ax.set_title('Influence vs. Error')
        ax.legend()

        # target vs. predicted (using ensemble) w/ uncertainty
        ax = axs[1][0]
        ax.errorbar(targets, pred_mean, yerr=pred_var, fmt='.', capsize=2, color='k', elinewidth=1)
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')

        title = f'{args.dataset.capitalize()}'
        if args.var_method == 'ensemble':
            title += f' ({args.n_ensemble:,} models; {args.sub_frac2:.1f} frac.)'
        ax.set_title(title)

        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # error vs. uncertainty (using ensemble)
        ax = axs[1][1]
        abs_error = np.abs(targets - pred_mean)

        p1 = pearsonr(conf_arr, abs_error)[0]
        s1 = spearmanr(conf_arr, abs_error)[0]

        ax.scatter(abs_error, conf_arr, color='none', marker='o', edgecolor='black',
                   label=f'{p1:.3f} (P), {s1:.3f} (S)')
        ax.set_xlabel('Variance of prediction (uncertainty)')
        ax.set_ylabel('L1 error (absolute value)')
        ax.set_title('Uncertainty vs. Absolute Error')
        ax.legend()

        # uncertainty (using ensemble) vs. influence sum
        ax = axs[1][2]
        p1 = pearsonr(inf_sum_arr, conf_arr)[0]
        s1 = spearmanr(inf_sum_arr, conf_arr)[0]

        ax.scatter(inf_sum_arr, conf_arr, color='none', marker='o', edgecolor='black',
                   label=f'{p1:.3f} (P), {s1:.3f} (S)')
        ax.set_xlabel('Sum of influence values')
        ax.set_ylabel('Variance of prediction (uncertainty)')
        ax.set_title('Influence vs. Uncertainty')
        ax.legend()

        # save plot
        correlation_dir = os.path.join(out_dir, '..', 'correlation', objective, args.dataset, args.var_method)
        os.makedirs(correlation_dir, exist_ok=True)

        plt.tight_layout()
        plt.savefig(os.path.join(correlation_dir, f'{args.method}.png'), bbox_inches='tight')


def main(args):

    # get method params and unique settings hash
    _, hash_str = exp_util.explainer_params_to_dict(args.method, vars(args))

    # experiment hash_str
    exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
                'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
    exp_hash = exp_util.dict_to_hash(exp_dict)

    # method1 dir
    in_dir = os.path.join(args.in_dir,
                          args.dataset,
                          args.tree_type,
                          f'exp_{exp_hash}',
                          f'{args.method}_{hash_str}')

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.inf_obj,
                           args.dataset)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, in_dir, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_influence')
    parser.add_argument('--in_dir2', type=str, default='output/variance/')
    parser.add_argument('--out_dir', type=str, default='output/plot/confidence/')

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
    parser.add_argument('--method', type=str, default='boostin')
    parser.add_argument('--zoom', type=float, nargs='+', default=[1.0])

    parser.add_argument('--var_method', type=str, default='ensemble')
    parser.add_argument('--n_ensemble', type=int, default=10)
    parser.add_argument('--sub_frac2', type=float, default=0.7)

    args = parser.parse_args()
    main(args)
