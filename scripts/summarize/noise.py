"""
Aggregate results.
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
from experiments import util as exp_util
from postprocess import util as pp_util
from postprocess.leaf_analysis import filter_results
from config import summ_args


def process(args, out_dir, logger):

    # get dataset
    color, line, label = pp_util.get_plot_dicts()

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

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        args.dataset = dataset

        r1 = {}

        exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
        exp_hash = exp_util.dict_to_hash(exp_dict)

        for random_state in range(1, args.n_repeat + 1):
            r1[random_state] = {}

            for strategy in args.strategy:

                exp_dir = os.path.join(args.in_dir,
                                       args.dataset,
                                       args.tree_type,
                                       f'exp_{exp_hash}',
                                       strategy,
                                       f'random_state_{random_state}')

                res_list = pp_util.get_results(args, args.in_dir, exp_dir, logger, progress_bar=False)
                res_list = filter_results(res_list, args.skip)

                for method, d in res_list:
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
        temp_dict = {'dataset': dataset, 'noise_frac': args.noise_frac}

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

    logger.info(f'\n\nClean loss:\n{cl_df}')
    logger.info(f'\nNoise loss:\n{nl_df}')
    logger.info(f'\nFixed loss:\n{fl_df}')

    logger.info(f'\n\nClean acc.:\n{cac_df}')
    logger.info(f'\nNoise acc.:\n{nac_df}')
    logger.info(f'\nFixed acc.:\n{fac_df}')

    logger.info(f'\n\nClean AUC:\n{cau_df}')
    logger.info(f'\nNoise AUC:\n{nau_df}')
    logger.info(f'\nFixed AUC:\n{fau_df}')

    logger.info(f'\nSaving results to {out_dir}...')

    fd_df.to_csv(os.path.join(out_dir, 'frac_detected.csv'), index=None)
    fd2_df.to_csv(os.path.join(out_dir, 'frac_detected_sem.csv'), index=None)

    fl_df.to_csv(os.path.join(out_dir, 'loss.csv'), index=None)
    fl2_df.to_csv(os.path.join(out_dir, 'loss_sem.csv'), index=None)

    fac_df.to_csv(os.path.join(out_dir, 'acc.csv'), index=None)
    fac2_df.to_csv(os.path.join(out_dir, 'acc_sem.csv'), index=None)

    fau_df.to_csv(os.path.join(out_dir, 'auc.csv'), index=None)
    fau_df.to_csv(os.path.join(out_dir, 'auc_sem.csv'), index=None)


def main(args):

    args.method += ['loss']

    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           f'noise{args.noise_frac}_check{args.check_frac}',
                           'summary')

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, out_dir, logger)


if __name__ == '__main__':
    main(summ_args.get_noise_args().parse_args())
