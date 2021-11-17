"""
Rank summarization results.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import plot_args


def get_results(in_dir_list, order, fn='loss_rank.csv'):
    """
    Read and order results for each experiment.

    Input
        in_dir_list: list, experiment output rank directories.
        order: list, order to put the methods in.
        fn: str, result file to read for each experiment.

    Return
        Mean and standard error dataframes covering all experiments/methods.
    """
    mean_dict = {}
    sem_dict = {}

    # get average rankings and relative performances from each scenario
    for name, in_dir in in_dir_list:
        fp = os.path.join(in_dir, fn)
        assert os.path.exists(fp), f'{fp} does not exist!'

        df = pd.read_csv(fp, header=0, names=['index', 'mean', 'sem'])
        df = df.sort_values('index')

        if name == 'Fix mislabeled':
            df = df[df['index'] != 'Loss_self'].copy()
            df['index'] = df['index'].apply(lambda x: x.replace('_test_sum', ''))

        df['index'] = df['index'].apply(lambda x: 'SubSample' if x == 'SubS.' else x)
        df.index = df['index']

        df = df.rename(index={'RandomSL': 'Target'})
        df = df.loc[order]

        mean_dict[name] = df['mean'].values
        sem_dict[name] = df['sem'].values

    mean_df = pd.DataFrame(mean_dict, index=df['index']).transpose()
    sem_df = pd.DataFrame(sem_dict, index=df['index']).transpose()

    # rename columns
    mean_df = mean_df.rename(columns={'Target': 'RandomSL'})
    sem_df = sem_df.rename(columns={'Target': 'RandomSL'})

    return mean_df, sem_df


def set_bars(ax):
    """
    Individually set the design of each bar.
    """

    # # hatches
    # h1 = '/' * 5
    # h2 = '.' * 3

    # bars = ax.patches
    # patterns = (h1, h2, h1, h2, h1, h2, h1, h2, h1, h2)
    # hatches = [p for p in patterns for i in range(4)]

    for bar in ax.patches:
        bar.set_edgecolor('k')
        bar.set_linewidth(0.75)
        bar.set_alpha(0.75)


def process(args, in_dir_list, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    order1 = ['Random', 'Target', 'TreeSim', 'BoostIn', 'BoostInW1', 'BoostInW2'
              'LeafInfSP', 'TREX', 'SubSample', 'LOO']
    order2 = ['BoostIn', 'BoostInW1', 'BoostInW2', 'LeafInfSP', 'TreeSim',
              'TREX', 'SubSample', 'LOO', 'Target', 'Random']

    rank_mean_df, rank_sem_df = get_results(in_dir_list, order2, fn='loss_rank.csv')
    rel_mean_df, rel_sem_df = get_results(in_dir_list, order2, fn='loss_rel.csv')

    # rel_mean_df = rel_mean_df.drop('Random', axis=1)
    # rel_sem_df = rel_sem_df.drop('Random', axis=1)

    # # get average rankings and relative performances from each scenario
    # for name, in_dir in in_dir_list:
    #     fp = os.path.join(in_dir, 'loss_rank.csv')
    #     assert os.path.exists(fp), f'{fp} does not exist!'

    #     df = pd.read_csv(fp, header=0, names=['index', 'mean', 'sem'])
    #     df = df.sort_values('index')

    #     if name == 'Fix mislabeled':
    #         df = df[df['index'] != 'Loss_self'].copy()
    #         df['index'] = df['index'].apply(lambda x: x.replace('_test_sum', ''))

    #     df['index'] = df['index'].apply(lambda x: 'SubSample' if x == 'SubS.' else x)
    #     df.index = df['index']

    #     df = df.rename(index={'RandomSL': 'Target'})
    #     df = df.loc[order2]

    #     mean_dict[name] = df['mean'].values
    #     sem_dict[name] = df['sem'].values

    # mean_df = pd.DataFrame(mean_dict, index=df['index']).transpose()
    # sem_df = pd.DataFrame(sem_dict, index=df['index']).transpose()

    # # rename columns
    # mean_df = mean_df.rename(columns={'Target': 'RandomSL'})
    # sem_df = sem_df.rename(columns={'Target': 'RandomSL'})

    # plot
    pp_util.plot_settings(fontsize=13)
    plt.rcParams.update({'hatch.color': 'k'})

    # fig, ax = plt.subplots(figsize=(9, 2.5))
    fig, axs = plt.subplots(2, 1, figsize=(9, 5))
    axs = axs.flatten()

    # ranking
    ax = axs[0]
    rank_mean_df.plot(kind='bar', yerr=rank_sem_df, ax=ax, rot=0, width=0.75, colormap='gnuplot2',
                      capsize=3, ylabel='Average rank', xlabel=None)
    set_bars(ax)
    ax.set_xticklabels([])
    # ax.axvline(1.5, color='k', linestyle='--')
    # ax.set_title('  Single Test Instance                         Multiple Test Instances')

    # relative performance
    ax = axs[1]
    rel_mean_df.plot(kind='bar', yerr=None, ax=ax, rot=0, width=0.75, colormap='gnuplot2',
                     capsize=3, ylabel='Geo. avg. loss increase\n(relative to Random)',
                     xlabel='Evaluation Setting', legend=None)
    set_bars(ax)
    ax.set_ylim(1.0, None)
    # print(dir(ax.get_yticklabels()[0]))
    # ax.set_yticklabels([y._text + 'x' for y in ax.get_yticklabels()])
    # ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    ticks_y = ticker.FuncFormatter(lambda x, pos: f'{x}x')
    ax.yaxis.set_major_formatter(ticks_y)
    # ax.axvline(1.5, color='k', linestyle='--')
    # ax.set_title('  Single Test Instance                         Multiple Test Instances')

    # hatches
    # h1 = '/' * 5
    # h2 = '.' * 3

    # bars = ax.patches
    # patterns = (h1, h2, h1, h2, h1, h2, h1, h2, h1, h2)
    # hatches = [p for p in patterns for i in range(4)]
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_edgecolor('k')
    #     bar.set_linewidth(0.75)
    #     bar.set_alpha(0.75)

    # plt.rcParams.update({'hatch.color': 'k'})

    # # legend
    # ax.legend(bbox_to_anchor=(0.44, 1.2275), loc='upper center',
    #           ncol=int(len(order2) / 2), framealpha=1.0, fontsize=11)

    # legend
    axs[0].legend(bbox_to_anchor=(0.5, 1.5), loc='upper center',
                  ncol=int(len(order2) / 2), framealpha=1.0, fontsize=11)

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.savefig(os.path.join(out_dir, 'rank.pdf'), bbox_inches='tight')
    plt.tight_layout()

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    assert len(args.tree_type) > 0
    tree_types = '+'.join(args.tree_type)

    remove_hash = exp_util.dict_to_hash({'n_test': args.n_test, 'remove_frac': args.remove_frac})
    remove_dir = os.path.join(args.in_dir, 'remove', 'rank', f'exp_{remove_hash}', tree_types)

    label_hash = exp_util.dict_to_hash({'n_test': args.n_test, 'edit_frac': args.edit_frac})
    label_dir = os.path.join(args.in_dir, 'label', 'rank', f'exp_{label_hash}', tree_types)

    remove_set_hash = exp_util.dict_to_hash({'val_frac': args.val_frac, 'remove_frac': args.remove_set_frac})
    remove_set_dir = os.path.join(args.in_dir, 'remove_set', 'rank', f'exp_{remove_set_hash}', tree_types)

    label_set_hash = exp_util.dict_to_hash({'val_frac': args.val_frac, 'edit_frac': args.edit_set_frac})
    label_set_dir = os.path.join(args.in_dir, 'label_set', 'rank', f'exp_{label_set_hash}', tree_types)

    # noise_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
    # noise_hash = exp_util.dict_to_hash(noise_dict)
    # noise_fp = os.path.join(args.in_dir, 'noise', 'rank', f'exp_{noise_hash}', tree_types, 'loss_rank.csv')

    single_list = [('Remove', remove_dir), ('Relabel', label_dir)]
    multi_list = [('Remove (set)', remove_set_dir), ('Relabel (set)', label_set_dir)]

    in_dir_list = single_list
    in_dir_list = multi_list

    out_dir = os.path.join(args.in_dir, 'ranking', tree_types)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, in_dir_list, out_dir, logger)


if __name__ == '__main__':
    main(plot_args.get_ranking_args().parse_args())
