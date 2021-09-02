"""
Aggregate results and organize them into one dict.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util as pp_util
from experiments import util
from leaf_analysis import filter_results


def process(args, out_dir, logger):

    # get dataset
    color, line, label = pp_util.get_plot_dicts()

    results = []

    r1 = {}

    exp_dict = {'noise': args.noise, 'noise_frac': args.noise_frac,
                'val_frac': args.val_frac, 'check_frac': args.check_frac}
    exp_hash = util.dict_to_hash(exp_dict)

    for random_state in args.random_state:
        r1[random_state] = {}

        for strategy in args.strategy:

            exp_dir = os.path.join(args.in_dir,
                                   args.dataset,
                                   args.tree_type,
                                   f'exp_{exp_hash}',
                                   strategy,
                                   f'random_state_{random_state}')

            res = filter_results(pp_util.get_results(args, args.in_dir, logger, exp_hash=exp_hash,
                                                     temp_dir=exp_dir, progress_bar=False), args.skip)

            for method, d in res:
                noise_idxs = d['noise_idxs']
                check_idxs = d['check_idxs']

                n_detected = len(set(noise_idxs).intersection(set(check_idxs)))
                frac_detected = n_detected / len(check_idxs)
                overall_frac_detected = n_detected / len(noise_idxs)

                res_simple = {'n_detected': n_detected, 'frac_detected': frac_detected}
                res_simple['clean_loss'] = d['res_clean']['loss']
                res_simple['clean_acc'] = d['res_clean']['acc']
                res_simple['clean_auc'] = d['res_clean']['auc']
                res_simple['noise_loss'] = d['res_noise']['loss']
                res_simple['noise_acc'] = d['res_noise']['acc']
                res_simple['noise_auc'] = d['res_noise']['auc']
                res_simple['fixed_loss'] = d['res_fixed']['loss']
                res_simple['fixed_acc'] = d['res_fixed']['acc']
                res_simple['fixed_auc'] = d['res_fixed']['auc']

                name = f'{label[method]}_{strategy}'

                r1[random_state][name] = res_simple

    # average over random states
    nd_list = []
    fd_list, fd2_list = [], []

    cl_list, cl2_list = [], []
    nl_list, nl2_list = [], []
    fl_list, fl2_list = [], []

    cac_list, cac2_list = [], []
    nac_list, nac2_list = [], []
    fac_list, fac2_list = [], []

    cau_list, cau2_list = [], []
    nau_list, nau2_list = [], []
    fau_list, fau2_list = [], []

    temp_dict = {'noise_frac': args.noise_frac}

    nd = temp_dict.copy()  # no. runs
    fd, fd2 = temp_dict.copy(), temp_dict.copy()  # frac_detected
    cl, cl2 = temp_dict.copy(), temp_dict.copy()  # clean_loss
    nl, nl2 = temp_dict.copy(), temp_dict.copy()  # noise_loss
    fl, fl2 = temp_dict.copy(), temp_dict.copy()  # fixed_loss
    cac, cac2 = temp_dict.copy(), temp_dict.copy()  # clean_acc
    nac, nac2 = temp_dict.copy(), temp_dict.copy()  # noise_acc
    fac, fac2 = temp_dict.copy(), temp_dict.copy()  # fixed_acc
    cau, cau2 = temp_dict.copy(), temp_dict.copy()  # clean_auc
    nau, nau2 = temp_dict.copy(), temp_dict.copy()  # noise_auc
    fau, fau2 = temp_dict.copy(), temp_dict.copy()  # fixed_auc

    for name in r1[random_state].keys():

        frac_detected_vals = [r1[rs][name]['frac_detected'] for rs in r1.keys()]
        fd[name] = np.mean(frac_detected_vals)
        fd2[name] = sem(frac_detected_vals)

        nd[name] = len(frac_detected_vals)

        # loss
        clean_loss_vals = [r1[rs][name]['clean_loss'] for rs in r1.keys()]
        cl[name] = np.mean(clean_loss_vals)
        cl2[name] = sem(clean_loss_vals)

        noise_loss_vals = [r1[rs][name]['noise_loss'] for rs in r1.keys()]
        nl[name] = np.mean(noise_loss_vals)
        nl2[name] = sem(noise_loss_vals)

        fixed_loss_vals = [r1[rs][name]['fixed_loss'] for rs in r1.keys()]
        fl[name] = np.mean(fixed_loss_vals)
        fl2[name] = sem(fixed_loss_vals)

        # acc.
        clean_acc_vals = [r1[rs][name]['clean_acc'] for rs in r1.keys()]
        cac[name] = np.mean(clean_acc_vals)
        cac2[name] = sem(clean_acc_vals)

        noise_acc_vals = [r1[rs][name]['noise_acc'] for rs in r1.keys()]
        nac[name] = np.mean(noise_acc_vals)
        nac2[name] = sem(noise_acc_vals)

        fixed_acc_vals = [r1[rs][name]['fixed_acc'] for rs in r1.keys()]
        fac[name] = np.mean(fixed_acc_vals)
        fac2[name] = sem(fixed_acc_vals)

        # AUC
        clean_auc_vals = [r1[rs][name]['clean_auc'] for rs in r1.keys()]
        cau[name] = np.mean(clean_auc_vals)
        cau2[name] = sem(clean_auc_vals)

        noise_auc_vals = [r1[rs][name]['noise_auc'] for rs in r1.keys()]
        nau[name] = np.mean(noise_auc_vals)
        nau2[name] = sem(noise_auc_vals)

        fixed_auc_vals = [r1[rs][name]['fixed_auc'] for rs in r1.keys()]
        fau[name] = np.mean(fixed_auc_vals)
        fau2[name] = sem(fixed_auc_vals)

    # compile results
    nd_list.append(nd)

    fd_list.append(fd)
    fd2_list.append(fd2)

    cl_list.append(cl)
    cl2_list.append(cl2)

    nl_list.append(nl)
    nl2_list.append(nl2)

    fl_list.append(fl)
    fl2_list.append(fl2)

    cac_list.append(cac)
    cac2_list.append(cac2)

    nac_list.append(nac)
    nac2_list.append(nac2)

    fac_list.append(fac)
    fac2_list.append(fac2)

    cau_list.append(cau)
    cau2_list.append(cau2)

    nau_list.append(nau)
    nau2_list.append(nau2)

    fau_list.append(fau)
    fau2_list.append(fau2)

    # organize results
    nd_df = pd.DataFrame(nd_list)
    fd_df, fd2_df = pd.DataFrame(fd_list), pd.DataFrame(fd2_list)
    cl_df, cl2_df = pd.DataFrame(cl_list), pd.DataFrame(cl2_list)
    nl_df, nl2_df = pd.DataFrame(nl_list), pd.DataFrame(nl2_list)
    fl_df, fl2_df = pd.DataFrame(fl_list), pd.DataFrame(fl2_list)
    cac_df, cac2_df = pd.DataFrame(cac_list), pd.DataFrame(cac2_list)
    nac_df, nac2_df = pd.DataFrame(nac_list), pd.DataFrame(nac2_list)
    fac_df, fac2_df = pd.DataFrame(fac_list), pd.DataFrame(fac2_list)
    cau_df, cau2_df = pd.DataFrame(cau_list), pd.DataFrame(cau2_list)
    nau_df, nau2_df = pd.DataFrame(nau_list), pd.DataFrame(nau2_list)
    fau_df, fau2_df = pd.DataFrame(fau_list), pd.DataFrame(fau2_list)

    logger.info(f'\nNo. runs:\n{nd_df}')

    logger.info(f'\n\nFrac. detected:\n{fd_df}\nFrac. detected (std. error):\n{fd2_df}')

    logger.info(f'\n\nClean loss:\n{cl_df}\nClean loss (std. error):\n{cl2_df}')
    logger.info(f'\nNoise loss:\n{nl_df}\nNoise loss (std. error):\n{nl2_df}')
    logger.info(f'\nFixed loss:\n{fl_df}\nFixed loss (std. error):\n{fl2_df}')

    logger.info(f'\n\nClean acc.:\n{cac_df}\nClean acc. (std. error):\n{cac2_df}')
    logger.info(f'\nNoise acc.:\n{nac_df}\nNoise acc. (std. error):\n{nac2_df}')
    logger.info(f'\nFixed acc.:\n{fac_df}\nFixed acc. (std. error):\n{fac2_df}')

    logger.info(f'\n\nClean AUC:\n{cau_df}\nClean AUC (std. error):\n{cau2_df}')
    logger.info(f'\nNoise AUC:\n{nau_df}\nNoise AUC (std. error):\n{nau2_df}')
    logger.info(f'\nFixed AUC:\n{fau_df}\nFixed AUC (std. error):\n{fau2_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    ax = axs[0]
    ax.errorbar(x=fd_df.columns.tolist()[1:], y=fd_df.iloc[0].values[1:] * 100,
                yerr=fd2_df.iloc[0].values[1:] * 100, linestyle='', marker='o', capsize=2.5, color='k')
    ax.set_ylabel('% noisy examples detected')
    ax.set_title(f'Recall ({args.noise_frac * 100:.0f}% noise, check {args.check_frac * 100:.0f}%)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax = axs[1]
    ax.errorbar(x=fl_df.columns.tolist()[1:], y=fl_df.iloc[0].values[1:],
                yerr=fl2_df.iloc[0].values[1:], linestyle='', marker='o', capsize=2.5, color='k')
    ax.axhline(cl_df.iloc[0].values[1], linestyle='--', color='k', label='Clean loss')
    ax.axhline(nl_df.iloc[0].values[1], linestyle=':', color='k', label='Noise loss')
    ax.set_ylabel('Test loss after fixing')
    ax.set_title(f'Test Loss')
    ax.legend(fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax = axs[2]
    ax.errorbar(x=fac_df.columns.tolist()[1:], y=fac_df.iloc[0].values[1:],
                yerr=fac2_df.iloc[0].values[1:], linestyle='', marker='o', capsize=2.5, color='k')
    ax.axhline(cac_df.iloc[0].values[1], linestyle='--', color='k', label='Clean acc.')
    ax.axhline(nac_df.iloc[0].values[1], linestyle=':', color='k', label='Noise acc.')
    ax.set_ylabel('Test accuracy after fixing')
    ax.set_title(f'Test Accuracy')
    ax.legend(fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax = axs[3]
    ax.errorbar(x=fau_df.columns.tolist()[1:], y=fau_df.iloc[0].values[1:],
                yerr=fau2_df.iloc[0].values[1:], linestyle='', marker='o', capsize=2.5, color='k')
    ax.axhline(cau_df.iloc[0].values[1], linestyle='--', color='k', label='Clean AUC')
    ax.axhline(nau_df.iloc[0].values[1], linestyle=':', color='k', label='Noise AUC')
    ax.set_ylabel('Test AUC after fixing')
    ax.set_title(f'Test AUC')
    ax.legend(fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{args.dataset}.png'), bbox_inches='tight')


def main(args):

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'noise{args.noise_frac}_check{args.check_frac}')

    log_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'noise{args.noise_frac}_check{args.check_frac}',
                           'logs')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = util.get_logger(os.path.join(log_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_noise/')
    parser.add_argument('--out_dir', type=str, default='output/plot/noise/')

    # experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--strategy', type=str, nargs='+', default=['self', 'test_sum'])
    parser.add_argument('--noise', type=str, default='target')
    parser.add_argument('--noise_frac', type=float, default=0.1)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--check_frac', type=float, default=0.1)

    # additional settings
    parser.add_argument('--random_state', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap

    # method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['loss', 'trex', 'similarity', 'similarity', 'boostin2',
                                 'leaf_influenceSP', 'subsample', 'loo', 'target', 'random'])
    parser.add_argument('--skip', type=str, nargs='+', default=[])
    parser.add_argument('--leaf_scale', type=int, nargs='+', default=[-1.0])  # BoostIn
    parser.add_argument('--local_op', type=str, nargs='+', default=['normal'])  # BoostIn
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

    args = parser.parse_args()
    main(args)
