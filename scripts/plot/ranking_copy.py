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

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from postprocess import util as pp_util
from experiments import util as exp_util
from config import plot_args


def process(args, fp_list, out_dir, logger):
    begin = time.time()

    color, line, label = pp_util.get_plot_dicts()

    mean_dict = {}
    sem_dict = {}

    order1 = ['Random', 'Target', 'TreeSim', 'BoostIn', 'LeafInfSP', 'TREX', 'SubSample', 'LOO']
    order2 = ['BoostIn', 'LeafInfSP', 'TreeSim', 'TREX', 'SubSample', 'LOO', 'Target', 'Random']

    # get average rankings from each scenario
    for name, fp in fp_list:
        assert os.path.exists(fp), f'{fp} does not exist!'
        df = pd.read_csv(fp, header=0, names=['index', 'mean', 'sem'])
        df = df.sort_values('index')

        if name == 'Fix mislabeled':
            df = df[df['index'] != 'Loss_self'].copy()
            df['index'] = df['index'].apply(lambda x: x.replace('_test_sum', ''))

        df['index'] = df['index'].apply(lambda x: 'SubSample' if x == 'SubS.' else x)
        df.index = df['index']

        df = df.rename(index={'RandomSL': 'Target'})
        df = df.loc[order2]

        mean_dict[name] = df['mean'].values
        sem_dict[name] = df['sem'].values

    mean_df = pd.DataFrame(mean_dict, index=df['index']).transpose()
    sem_df = pd.DataFrame(sem_dict, index=df['index']).transpose()

    # rename columns
    mean_df = mean_df.rename(columns={'Target': 'RandomSL'})
    sem_df = sem_df.rename(columns={'Target': 'RandomSL'})

    # plot
    pp_util.plot_settings(fontsize=13)

    fig, ax = plt.subplots(figsize=(9, 2.5))

    mean_df.plot(kind='bar', yerr=sem_df, ax=ax, rot=0, width=0.75, colormap='gnuplot2',
                 capsize=3, ylabel='Average rank', xlabel='Evaluation Setting')

    ax.axvline(1.5, color='k', linestyle='--')

    ax.set_title('  Single Test Instance                         Multiple Test Instances')

    # hatches
    h1 = '/' * 5
    h2 = '.' * 3

    bars = ax.patches
    patterns = (h1, h2, h1, h2, h1, h2, h1, h2)
    hatches = [p for p in patterns for i in range(4)]
    for bar, hatch in zip(bars, hatches):
        bar.set_edgecolor('k')
        bar.set_linewidth(0.75)
        bar.set_alpha(0.75)

    plt.rcParams.update({'hatch.color': 'k'})

    # # legend
    # ax.legend(bbox_to_anchor=(0.44, 1.2275), loc='upper center',
    #           ncol=int(len(order2) / 2), framealpha=1.0, fontsize=11)

    # legend
    ax.legend(bbox_to_anchor=(0.5, 1.5), loc='upper center',
              ncol=int(len(order2) / 2), framealpha=1.0, fontsize=11)

    logger.info(f'\nSaving results to {out_dir}/...')

    plt.savefig(os.path.join(out_dir, 'rank.pdf'), bbox_inches='tight')
    plt.tight_layout()

    logger.info(f'\nTotal time: {time.time() - begin:.3f}s')


def main(args):

    assert len(args.tree_type) > 0
    tree_types = '+'.join(args.tree_type)

    roar_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}
    roar_hash = exp_util.dict_to_hash(roar_dict)
    roar_fp = os.path.join(args.in_dir, 'roar', 'rank', f'exp_{roar_hash}', tree_types, 'loss_rank.csv')

    cf_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac, 'step_size': args.step_size}
    cf_hash = exp_util.dict_to_hash(cf_dict)
    cf_fp = os.path.join(args.in_dir, 'counterfactual', 'rank', f'exp_{cf_hash}', tree_types, 'frac_edits_rank.csv')

    poison_dict = {'poison_frac': args.poison_frac, 'val_frac': args.val_frac}
    poison_hash = exp_util.dict_to_hash(poison_dict)
    poison_fp = os.path.join(args.in_dir, 'poison', 'rank', f'exp_{poison_hash}', tree_types, 'loss_rank.csv')

    noise_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac, 'check_frac': args.check_frac}
    noise_hash = exp_util.dict_to_hash(noise_dict)
    noise_fp = os.path.join(args.in_dir, 'noise', 'rank', f'exp_{noise_hash}', tree_types, 'loss_rank.csv')

    fp_list = [('ROAR', roar_fp), ('Counterfactual', cf_fp), ('Poison', poison_fp), ('Fix mislabeled', noise_fp)]

    out_dir = os.path.join(args.in_dir, 'ranking', tree_types)

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)

    logger = exp_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    process(args, fp_list, out_dir, logger)


if __name__ == '__main__':
    main(plot_args.get_ranking_args().parse_args())
