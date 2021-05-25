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
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from utility import print_util


def get_result(template, in_dir):
    """
    Obtain the results for this baseline method.
    """
    result = template.copy()

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]
        result.update(d)

    return result


def process_utility(gf):
    """
    Processes utility differences BEFORE deletion,
    and averages the results over different random states.
    """
    result = {}

    model_acc_list = []
    model_auc_list = []
    model_ap_list = []

    acc_diff_list = []
    auc_diff_list = []
    ap_diff_list = []

    model_delete_time_list = []

    for row in gf.itertuples(index=False):

        # extract model predictive performance
        model_acc_list.append(row.model_acc)
        model_auc_list.append(row.model_auc)
        model_ap_list.append(row.model_ap)

        # compare model predictive performance to naive
        acc_diff_list.append(row.naive_acc - row.model_acc)
        auc_diff_list.append(row.naive_auc - row.model_auc)
        ap_diff_list.append(row.naive_ap - row.model_ap)

        # record avg. deletion time for the model
        model_delete_time_list.append(row.naive_avg_delete_time / row.model_n_deleted)

    # compute mean and sem for predictive performances
    result['model_acc_mean'] = np.mean(model_acc_list)
    result['model_auc_mean'] = np.mean(model_auc_list)
    result['model_ap_mean'] = np.mean(model_ap_list)
    result['model_acc_sem'] = sem(model_acc_list)
    result['model_auc_sem'] = sem(model_auc_list)
    result['model_ap_sem'] = sem(model_ap_list)
    result['model_delete_time_mean'] = np.mean(model_delete_time_list)
    result['model_delete_time_sem'] = sem(model_delete_time_list)

    # compute mean and sem for predictive performance differences
    result['acc_diff_mean'] = np.mean(acc_diff_list)
    result['auc_diff_mean'] = np.mean(auc_diff_list)
    result['ap_diff_mean'] = np.mean(ap_diff_list)
    result['acc_diff_sem'] = sem(acc_diff_list)
    result['auc_diff_sem'] = sem(auc_diff_list)
    result['ap_diff_sem'] = sem(ap_diff_list)

    return result


def process_retrains(gf, max_depth=20):
    """
    Averages no. retrains and retrain costs over multiple runs for each depth.
    """
    n_retrains = np.zeros(shape=(len(gf), max_depth))
    retrain_costs = np.zeros(shape=(len(gf), max_depth))

    i = 0
    for row in gf.itertuples(index=False):

        for j in range(max_depth):

            # add deletions to this depth
            if 1 in row.model_delete_depths and j in row.model_delete_depths[1]:
                n_retrains[i][j] = row.model_delete_depths[1][j]

            if j in row.model_delete_costs:
                retrain_costs[i][j] = row.model_delete_costs[j]

    # compute average no. rertains and retrain costs for each depths
    n_retrains_mean = np.mean(n_retrains, axis=0)
    retrain_costs_mean = np.mean(retrain_costs, axis=0)

    # build results
    n_retrains_result = {k: v for k, v in zip(range(n_retrains_mean.shape[0]), n_retrains_mean)}
    retrain_costs_result = {k: v for k, v in zip(range(retrain_costs_mean.shape[0]), retrain_costs_mean)}

    return n_retrains_result, retrain_costs_result


def process_results(df):
    """
    Processes utility differences, retrains, and averages the results
    over different random states.
    """
    setting_cols = ['dataset', 'criterion', 'n_estimators', 'max_depth',
                    'topd', 'k', 'subsample_size']

    keep_cols = ['naive_avg_delete_time',
                 'naive_n_deleted',
                 'model_n_deleted',
                 'model_train_%_deleted',
                 'model_n_nodes_avg',
                 'model_n_random_nodes_avg',
                 'model_n_greedy_nodes_avg']

    # result containers
    main_result_list = []
    n_retrain_result_list = []
    retrain_cost_result_list = []

    # loop through each experiment setting
    i = 0
    for tup, gf in tqdm(df.groupby(setting_cols)):

        # create main result
        main_result = {k: v for k, v in zip(setting_cols, tup)}
        main_result['id'] = i
        main_result.update(process_utility(gf))
        for c in keep_cols:
            main_result[c] = gf[c].mean()
            main_result['{}_std'.format(c)] = gf[c].std()
        main_result_list.append(main_result)

        # process retrain results
        n_retrain_result, retrain_cost_result = process_retrains(gf)

        # create no. retrain result
        n_retrain_result['id'] = i
        n_retrain_result_list.append(n_retrain_result)

        # create retrain cost result
        retrain_cost_result['id'] = i
        retrain_cost_result_list.append(retrain_cost_result)
        i += 1

    # compile results
    main_df = pd.DataFrame(main_result_list)
    n_retrain_df = pd.DataFrame(n_retrain_result_list)
    retrain_cost_df = pd.DataFrame(retrain_cost_result_list)

    return main_df, n_retrain_df, retrain_cost_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset, args.criterion, args.rs,
                                         args.topd, args.k, args.subsample_size]))

    # cedar_settings = list(product(*[args.epsilon, args.lmbda]))

    results = []
    for dataset, criterion, rs, topd, k, sub_size in tqdm(experiment_settings):

        template = {'dataset': dataset,
                    'criterion': criterion,
                    'rs': rs,
                    'topd': topd,
                    'k': k,
                    'subsample_size': sub_size}

        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      criterion,
                                      'rs_{}'.format(rs),
                                      'topd_{}'.format(topd),
                                      'k_{}'.format(k),
                                      'sub_{}'.format(sub_size))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        result = get_result(template, experiment_dir)
        if result is not None:
            results.append(result)

    # display more columns
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    # collect raw results
    df = pd.DataFrame(results)
    logger.info('\nRaw results:\n{}'.format(df))

    # process results
    logger.info('\nProcessing results...')
    main_df, n_retrain_df, retrain_cost_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))
    logger.info('\nNo. retrain results:\n{}'.format(n_retrain_df))
    logger.info('\nRetrain cost results:\n{}'.format(retrain_cost_df))

    # create filepaths
    main_fp = os.path.join(out_dir, 'results.csv')
    n_retrain_fp = os.path.join(out_dir, 'n_retrain.csv')
    retrain_cost_fp = os.path.join(out_dir, 'retrain_cost.csv')

    # save processed results
    main_df.to_csv(main_fp, index=None)
    n_retrain_df.to_csv(n_retrain_fp, index=None)
    retrain_cost_df.to_csv(retrain_cost_fp, index=None)


def main(args):

    out_dir = os.path.join(args.out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    create_csv(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/delete/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'vaccine', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'census', 'credit_card', 'no_show', 'olympics', 'twitter', 'synthetic',
                                 'higgs', 'ctr'], help='dataset.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--subsample_size', type=int, nargs='+', default=[1, 1000], help='subsampling size.')

    # hyperparameter settings
    parser.add_argument('--topd', type=int, nargs='+', default=list(range(21)), help='top d.')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 5, 10, 25, 50, 100], help='no. thresholds.')

    args = parser.parse_args()
    main(args)
