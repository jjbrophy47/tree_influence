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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import plot_args


def get_results(in_dir_list, order, suffix='', fn='loss_rank.csv'):
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

        if name.startswith('Fix mislabelled'):
            fn = f'fd_rank{suffix}.csv'

        fp = os.path.join(in_dir, fn)
        assert os.path.exists(fp), f'{fp} does not exist!'

        df = pd.read_csv(fp, header=0, names=['index', 'mean', 'sem'])
        df = df.sort_values('index')

        # use consistent method names
        if name.startswith('Fix mislabelled'):
            df = df[df['index'] != 'Loss_self'].copy()
            df['index'] = df['index'].apply(lambda x: x.replace('_test_sum', ''))

        elif name.startswith('Targeted edit'):
            df['index'] = df['index'].apply(lambda x: x.replace('LE', ''))

        df['index'] = df['index'].apply(lambda x: 'SubSample' if x == 'SubS.' else x)
        df['index'] = df['index'].apply(lambda x: 'LeafInfluence' if x == 'LeafInf.' else x)
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


def set_hatches(bars, hatch='///'):
    """
    Individually set the design of each bar.

    Input
        bars: Bar container object returned from ax.bar method
    """
    for bar in bars:
        bar.set_edgecolor('k')
        bar.set_linewidth(0.75)
        bar.set_alpha(0.75)


def process(args, in_dir_list, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    if args.li:
        suffix = '_li'
        order = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim',
                 'LeafRefit', 'LeafInfluence', 'SubSample', 'LOO']

    else:
        pass
        suffix = ''
        order = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim', 'SubSample', 'LOO']

    rank_mean_df, rank_sem_df = get_results(in_dir_list, order, suffix=suffix, fn=f'{args.metric}_rank{suffix}.csv')

    fast_methods = ['BoostIn', 'LeafInfSP', 'TREX', 'TreeSim']
    slow_methods = ['LeafRefit', 'LeafInfluence', 'SubSample', 'Loo']

    # plot
    pp_util.plot_settings(fontsize=11)
    plt.rcParams.update({'hatch.color': 'lightgray'})

    fig, ax = plt.subplots(figsize=(9, 2.5))

    X = np.arange(len(in_dir_list))
    n_bars = len(order)
    total_width = 0.8
    width = total_width / n_bars
    separator = 0.025
    cmap = matplotlib.cm.get_cmap('gnuplot2')

    fast_bars = []
    fast_names = []

    slow_bars = []
    slow_names = []

    for i, col in enumerate(order):
        x_offset = (i - n_bars / 2) * width + width / 2

        if col in fast_methods:
            x_offset -= separator
            hatch = None

        else:
            x_offset += separator
            hatch = '//'

        color_frac = 0.0 if i == 0 else float(i) / n_bars

        bar = ax.bar(x=X + x_offset, height=rank_mean_df[col].values, yerr=rank_sem_df[col].values,
                     color=cmap(color_frac), width=width, capsize=3, hatch=hatch)

        if col in fast_methods:
            fast_bars.append(bar)
            fast_names.append(col)

        else:
            slow_bars.append(bar)
            slow_names.append(col)

    # plot loss_self frac. detected for fixing mislabelled
    fp = os.path.join(in_dir_list[-1][1], f'fd_rank{suffix}.csv')
    df = pd.read_csv(fp, header=0, names=['index', 'mean', 'sem'])
    df = df.sort_values('index')
    loss_row = df[df['index'] == 'Loss_self']
    loss_mean = loss_row.iloc[0]['mean']
    loss_sem = loss_row.iloc[0]['sem']

    x_offset = (len(order) - n_bars / 2) * width + width / 2 + (separator * 3)
    bar = ax.bar(x=len(X) - 1 + x_offset, height=loss_mean, yerr=loss_sem,
                 color=cmap(1.0), width=width, capsize=3, hatch='..', edgecolor='k')

    ax.set_ylabel('Average rank')
    ax.set_ylim(1.0, None)
    ax.set_xlabel('Evaluation setting')
    ax.set_xticks(X)
    ax.set_xticklabels([name for name, _ in in_dir_list])
    
    ax.axvline(1.5, linestyle='--', color='lightgray')
    ax.annotate('Single test', (0.01, 0.9), xycoords='axes fraction')
    ax.annotate('Multiple test', (0.41, 0.9), xycoords='axes fraction')

    # legend
    lg1 = plt.legend(handles=fast_bars, labels=fast_names, bbox_to_anchor=(0.25, 1.5),
                     loc='upper center', title='Fast Methods',
                     ncol=int(len(fast_names) / 2), framealpha=1.0, fontsize=11)

    lg2 = plt.legend(handles=slow_bars, labels=slow_names, bbox_to_anchor=(0.75, 1.5),
                     loc='upper center', title='Slow Methods',
                     ncol=int(len(slow_names) / 2), framealpha=1.0, fontsize=11)

    lg3 = plt.legend(handles=[bar], labels=['Loss'], bbox_to_anchor=(0.75, 0.99999),
                     loc='upper center', title=None,
                     ncol=1, framealpha=1.0, fontsize=11)

    ax.add_artist(lg1)
    ax.add_artist(lg2)

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.savefig(os.path.join(out_dir, f'rank_{args.metric}{suffix}.pdf'), bbox_inches='tight')
    plt.tight_layout()

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    assert len(args.tree_type) > 0
    tree_types = '+'.join(args.tree_type)

    remove_hash = exp_util.dict_to_hash({'n_test': args.n_test, 'remove_frac': args.remove_frac})
    remove_dir = os.path.join(args.in_dir, 'remove', 'rank', f'exp_{remove_hash}', tree_types)

    TEMP_TREE = 'lgb+sgb+xgb'
    targeted_edit_hash = exp_util.dict_to_hash({'n_test': args.n_test, 'edit_frac': args.edit_frac})
    targeted_edit_dir = os.path.join(args.in_dir, 'targeted_edit', 'rank', f'exp_{targeted_edit_hash}', tree_types)

    remove_set_hash = exp_util.dict_to_hash({'val_frac': args.val_frac, 'remove_frac': args.remove_set_frac})
    remove_set_dir = os.path.join(args.in_dir, 'remove_set', 'rank', f'exp_{remove_set_hash}', tree_types)

    label_set_hash = exp_util.dict_to_hash({'val_frac': args.val_frac, 'edit_frac': args.edit_set_frac})
    label_set_dir = os.path.join(args.in_dir, 'label_set', 'rank', f'exp_{label_set_hash}', tree_types)

    noise_set_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
    noise_set_hash = exp_util.dict_to_hash(noise_set_dict)
    noise_set_dir = os.path.join(args.in_dir, 'noise_set', 'rank', f'exp_{noise_set_hash}', tree_types)

    single_list = [('Removal', remove_dir), ('Targeted edit', targeted_edit_dir)]
    multi_list = [(' Removal ', remove_set_dir), ('Adding noise', label_set_dir),
                  ('Fix mislabelled', noise_set_dir)]

    in_dir_list = single_list + multi_list

    out_dir = os.path.join(args.in_dir, 'ranking', tree_types)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, in_dir_list, out_dir, logger)


if __name__ == '__main__':
    main(plot_args.get_ranking_args().parse_args())
