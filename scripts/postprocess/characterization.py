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


def get_cis(res):
    """
    Compute the min. frac. train examples that, when deleted,
        flip the prediction (counterfactual influence set).

    Input
        res: dict, ROAR result dictionary.

    Return
        - 1d array of CIS's of shape=(no. test,).
    """
    pred = res['pred']  # shape=(no. test, no. ckpts, no. class)
    frac_arr = res['remove_frac'][0]  # shape=(no. ckpts,)

    fracs = np.zeros(pred.shape[0], dtype=np.int32)  # shape=(no. test,)

    for i in range(idxs.shape[0]):

        if objective in ['binary', 'multiclass']:
            init_pred = np.argmax(pred[i, 0])  # pred. label at 1st ckpt.
            flip_idxs = np.where(np.argmax(pred[i], axis=1) != init_pred)[0]
            fracs[i] = frac_arr[-1] if len(flip_idxs) == 0 else frac_arr[flip_idxs[0]]

    return fracs


def experiment(args, logger, out_dir):

    # initialize experiment
    begin = time.time()

    # get dataset
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # get results
    results = pp_util.get_results(args, logger)
    color, line, label = pp_util.get_plot_dicts()

    if args.inf_obj == 'global':
        assert objective == 'binary'

        n_row = 2 if args.zoom > 0.0 and args.zoom < 1.0 else 1
        height = 8 if args.zoom > 0.0 and args.zoom < 1.0 else 4

        fig, axs = plt.subplots(n_row, 3, figsize=(12, height))
        axs = axs.flatten()

        frac_arr = np.linspace(0, 0.5, args.n_sample + 1)[1:]

        for method, res in results:

            inf = res['influence']
            pred = res['y_train_pred'].flatten()
            ranking = np.argsort(inf)[::-1]

            # result containers
            frac_pos_remove_arr = np.zeros(len(frac_arr), dtype=np.float32)
            frac_pos_total_arr = np.zeros(len(frac_arr), dtype=np.float32)
            loss_arr = np.zeros(len(frac_arr), dtype=np.float32)

            for i, frac in enumerate(frac_arr):
                n = int(len(inf) * frac)

                train_idxs = ranking[:n]
                frac_pos_remove_arr[i] = np.mean(y_train[train_idxs])
                frac_pos_total_arr[i] = np.sum(y_train[train_idxs]) / np.sum(y_train)
                loss_arr[i] = log_loss(y_train[train_idxs], pred[train_idxs], labels=[0, 1])

            # plot
            for i in range(3):
                ax = axs[i]

                if i == 0:
                    x, y = frac_arr * 100, frac_pos_remove_arr * 100
                    ax.plot(x, y, color=color[method], linestyle=line[method], label=label[method])
                    ax.set_xlabel('% train data removed')
                    ax.set_ylabel('% pos. examples in removed set')

                    if args.zoom > 0.0 and args.zoom < 1.0:
                        ax = axs[i + 3]
                        n = int(len(x) * args.zoom)
                        x, y = x[:n], y[:n]
                        ax.plot(x, y, color=color[method], linestyle=line[method], label=label[method])
                        ax.set_xlabel('% train data removed')
                        ax.set_ylabel('% pos. examples in removed set')

                elif i == 1:
                    x, y = frac_arr * 100, frac_pos_total_arr * 100
                    ax.plot(x, y, color=color[method],
                            linestyle=line[method], label=label[method])
                    ax.set_xlabel('% train data removed')
                    ax.set_ylabel('Overall % pos. examples removed')

                    if args.zoom > 0.0 and args.zoom < 1.0:
                        ax = axs[i + 3]
                        n = int(len(x) * args.zoom)
                        x, y = x[:n], y[:n]
                        ax.plot(x, y, color=color[method],
                                linestyle=line[method], label=label[method])
                        ax.set_xlabel('% train data removed')
                        ax.set_ylabel('Overall % pos. examples removed')

                elif i == 2:
                    x, y = frac_arr * 100, loss_arr
                    ax.plot(x, y, color=color[method], linestyle=line[method], label=label[method])
                    ax.set_xlabel('% train data removed')
                    ax.set_ylabel('Log loss of removed examples')
                    ax.legend(fontsize=6)

                    if args.zoom > 0.0 and args.zoom < 1.0:
                        ax = axs[i + 3]
                        n = int(len(x) * args.zoom)
                        x, y = x[:n], y[:n]
                        ax.plot(x, y, color=color[method], linestyle=line[method], label=label[method])
                        ax.set_xlabel('% train data removed')
                        ax.set_ylabel('Log loss of removed examples')

    # TODO: add support for multiclass
    # TODO: add support for regression
    else:  # local

        assert objective == 'binary'

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        
        for method, res in results:

            include = True
            for skip in args.skip:
                if skip in method:
                    include = False
                    break
            if not include:
                continue

            pred = res['pred']

            idxs = np.zeros(pred.shape[0], dtype=np.int32)
            for idx in range(idxs.shape[0]):
                init_pred = 1.0 if pred[idx][0] >= 0.5 else 0

                if init_pred == 0:
                    r = np.where(pred[idx] >= 0.5)[0]
                else:
                    r = np.where(pred[idx] < 0.5)[0]

                if len(r) == 0:
                    idxs[idx] = -1
                else:
                    idxs[idx] = r[0]


            fracs = res['remove_frac'][0][idxs] * 100
            fracs_zoom = fracs[np.where(fracs <= args.zoom)]  # cutoff

            f_mean, f_std = fracs.mean(), sem(fracs)
            fz_mean, fz_std = fracs_zoom.mean(), sem(fracs_zoom)

            sns.histplot(fracs, stat='count', cumulative=True, fill=False, element='step', ax=axs[0],
                         color=color[method], linestyle=line[method], alpha=0.65,
                         label=f'{label[method]}: {f_mean:.2f} +/- {f_std:.2f}')

            sns.histplot(fracs_zoom, stat='count', cumulative=True, fill=False, element='step', ax=axs[1],
                         color=color[method], linestyle=line[method], alpha=0.65,
                         label=f'{label[method]} ({len(fracs_zoom)}): {fz_mean:>5.2f} +/- {fz_std:>5.2f}')

        legend = axs[0].legend(fontsize=6, title='Avg. % Rem.')
        legend = axs[1].legend(fontsize=6, title='Avg. % Rem.')

        axs[0].set_xlabel('% train removed to flip pred.')
        axs[1].set_xlabel('% train removed to flip pred.')

        axs[0].set_ylabel('Cumulative no. test examples')
        axs[1].set_ylabel('Cumulative no. test examples')


    plt_dir = os.path.join(args.out_dir, args.inf_obj)
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
    parser.add_argument('--in_dir', type=str, default='output/influence/')
    parser.add_argument('--out_dir', type=str, default='output/plot/characterization/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='surgical')

    # Tree-ensemble settings
    parser.add_argument('--tree_type', type=str, default='lgb')

    # Method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['random', 'target', 'boostin', 'trex', 'similarity',
                                 'leaf_influence', 'loo', 'dshap'])
    parser.add_argument('--skip', type=str, nargs='+',
                        default=['minority', 'loss', 'boostin_9e', 'boostin_08'])
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
    parser.add_argument('--zoom', type=float, default=2.0)
    parser.add_argument('--n_sample', type=int, default=100)

    args = parser.parse_args()
    main(args)
