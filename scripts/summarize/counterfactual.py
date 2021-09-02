"""
Summarize counerfactual results.
"""
import os
import sys
import time
import hashlib
import argparse
import resource
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.metrics import log_loss

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util as pp_util
from experiments import util
from postprocess.leaf_analysis import filter_results


def process(args, logger, out_dir):
    begin = time.time()

    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    color, line, label = pp_util.get_plot_dicts()

    rows = []
    rows2 = []

    logger.info('')
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        args.dataset = dataset
        results = filter_results(pp_util.get_results(args, in_dir=args.in_dir,
                                                     logger=logger, progress_bar=False), args.skip)

        row = {'dataset': args.dataset}
        row2 = {'dataset': args.dataset}
        
        for i, (method, res) in enumerate(results):

            df = res['df']
            n_correct = len(df[df['status'] == 'success']) + len(df[df['status'] == 'fail'])

            # result containers
            frac_edits = np.full(n_correct, 0.1, dtype=np.float32)
            n_edits = np.full(n_correct, int(X_train.shape[0] * 0.1), dtype=np.float32)

            temp_df = df[df['status'] == 'success']

            if len(temp_df) > 0:
                frac_edits[:len(temp_df)] = temp_df['frac_edits'].values
                n_edits[:len(temp_df)] = temp_df['n_edits'].values

            row[label[method]] = np.mean(frac_edits)
            row2[label[method]] = sem(frac_edits)

        rows.append(row)
        rows2.append(row2)

    df = pd.DataFrame(rows)
    df2 = pd.DataFrame(rows2)

    logger.info(f'\nFrac. edit results:\n{df}')
    logger.info(f'\nFrac. edit std. error:\n{df2}')

    logger.info(f'\nSaving results to {out_dir}...')

    df.to_csv(os.path.join(out_dir, 'frac_edits.csv'), index=None)
    df2.to_csv(os.path.join(out_dir, 'frac_edits_sem.csv'), index=None)

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    # get method params and unique settings hash
    _, hash_str = util.explainer_params_to_dict(args.method, vars(args))

    # create output dir
    out_dir = os.path.join(args.out_dir,
                           args.tree_type,
                           'summary')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = util.get_logger(os.path.join(out_dir, f'{args.dataset}.txt'))
    logger.info(args)
    logger.info(datetime.now())

    process(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--in_dir', type=str, default='temp_counterfactual/')
    parser.add_argument('--out_dir', type=str, default='output/plot/counterfactual/')

    # experiment settings
    parser.add_argument('--dataset_list', type=str, nargs='+',
                        default=['adult', 'bank_marketing', 'bean', 'compas',
                                 'concrete', 'credit_card', 'diabetes', 'energy',
                                 'flight_delays', 'german_credit', 'htru2', 'life',
                                 'msd', 'naval', 'no_show', 'obesity', 'power', 'protein',
                                 'spambase', 'surgical', 'twitter', 'vaccine', 'wine'])
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
    parser.add_argument('--n_ckpt', type=int, default=50)

    # Method settings
    parser.add_argument('--method', type=str, nargs='+',
                        default=['trex', 'similarity', 'boostin2', 'leaf_influenceSP',
                                 'subsample', 'loo', 'target', 'random'])
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
