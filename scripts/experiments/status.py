"""
Show completed state for a given set of experiments.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util
from config import status_args


def get_experiment_hash(args):
    """
    Return the experiment hash for the given args.
    """

    # single test
    if args.exp == 'influence':
        exp_dict = {'n_test': args.n_test}

    elif args.exp == 'influenceLE':
        exp_dict = {'n_test': args.n_test}

    elif args.exp == 'remove':
        exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}

    elif args.exp == 'label':
        exp_dict = {'n_test': args.n_test, 'edit_frac': args.edit_frac}

    elif args.exp == 'poison':
        exp_dict = {'n_test': args.n_test, 'poison_frac': args.poison_frac}

    elif args.exp == 'counterfactual':
        exp_dict = {'n_test': args.n_test, 'step_size': args.step_size}

    elif args.exp == 'targeted_edit':
        exp_dict = {'n_test': args.n_test, 'edit_frac': args.targeted_edit_frac}

    # multi test
    elif args.exp == 'influence_set':
        exp_dict = {'val_frac': args.val_frac}

    elif args.exp == 'remove_set':
        exp_dict = {'remove_frac': args.remove_frac_set, 'val_frac': args.val_frac}

    elif args.exp == 'label_set':
        exp_dict = {'edit_frac': args.edit_frac_set, 'val_frac': args.val_frac}

    elif args.exp == 'poison_set':
        exp_dict = {'poison_frac': args.poison_frac_set, 'val_frac': args.val_frac}

    elif args.exp == 'noise_set':
        exp_dict = {'noise_frac': args.noise_frac, 'check_frac': args.check_frac,
                    'val_frac': args.val_frac}

    exp_hash = util.dict_to_hash(exp_dict)

    return exp_hash


def get_method_hash(args, method):
    """
    Get method hash for the given args for the specified method.
    """
    _, method_hash = util.explainer_params_to_dict(method, vars(args))
    return method_hash


def get_noise_set_status(args, logger, out_dir, exp_hash):
    """
    Construct pd.FataFrame with completion status of each experiment.

    Note
        - Custom method for the "Noise" experiment.
    """
    results = []
    
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        for tree_type in args.tree_type:
            logger.info(f'\t{tree_type}')

            method_results = {'dataset': dataset, 'tree_type': tree_type}

            for agg_type in args.agg_type:

                for method in args.method_list:

                    method_hash = get_method_hash(args, method)

                    result_dir = os.path.join(args.in_dir,
                                              dataset,
                                              tree_type,
                                              f'exp_{exp_hash}',
                                              agg_type,
                                              f'{method}_{method_hash}')

                    result_fp = os.path.join(result_dir, 'results.npy')

                    if os.path.exists(result_fp):

                        if args.status_type == 'time':
                            result = np.load(result_fp, allow_pickle=True)[()]
                            assert 'total_time' in result
                            method_results[f'{method}_{agg_type}'] = result['total_time']

                        elif args.status_type == 'completion':
                            method_results[f'{method}_{agg_type}'] = 1

            results.append(method_results)

    df = pd.DataFrame(results).sort_values(['tree_type', 'dataset'])
    logger.info(f'\nCompleted status:\n{df}')
    df.to_csv(os.path.join(out_dir, f'{args.status_type}.csv'), index=None)


def get_result_status(args, logger, out_dir, exp_hash):
    """
    Construct pd.FataFrame with completion status of each experiment.
    """

    results = []
    
    for dataset in args.dataset_list:
        logger.info(f'{dataset}')

        for tree_type in args.tree_type:
            logger.info(f'\t{tree_type}')

            method_results = {'dataset': dataset, 'tree_type': tree_type}

            for method in args.method_list:
                method_hash = get_method_hash(args, method)

                result_dir = os.path.join(args.in_dir,
                                          dataset,
                                          tree_type,
                                          f'exp_{exp_hash}',
                                          f'{method}_{method_hash}')

                result_fp = os.path.join(result_dir, 'results.npy')

                if os.path.exists(result_fp):

                    if args.status_type == 'time':
                        result = np.load(result_fp, allow_pickle=True)[()]
                        assert 'total_time' in result
                        method_results[method] = result['total_time']

                    elif args.status_type == 'completion':
                        method_results[method] = 1

                else:
                    method_results[method] = np.nan

            results.append(method_results)

    df = pd.DataFrame(results).sort_values(['tree_type', 'dataset'])
    logger.info(f'\nCompleted status:\n{df}')
    df.to_csv(os.path.join(out_dir, f'{args.status_type}.csv'), index=None)


def main(args):

    exp_hash = get_experiment_hash(args)

    # get status for specific methods
    if args.exp == 'noise_set':
        args.method_list = ['random', 'target', 'loss', 'leaf_sim', 'boostin',
                            'leaf_infSP', 'trex', 'subsample', 'loo', 'leaf_inf',
                            'leaf_refit', 'boostinW1', 'boostinW2']

    elif args.exp == 'targeted_edit':
        args.method_list = ['random', 'target', 'leaf_sim', 'trex',
                            'subsample', 'loo', 'leaf_refit', 'leaf_inf', 'leaf_infSP',
                            'boostin', 'boostinW1', 'boostinW2',
                            'boostinLE', 'boostinW1LE', 'boostinW2LE',
                            'looLE', 'leaf_refitLE', 'leaf_infLE', 'leaf_infSPLE']

    elif args.exp == 'influenceLE':
        args.method_list = ['boostinLE', 'boostinW1LE', 'boostinW2LE',
                            'leaf_infSPLE', 'looLE', 'leaf_refitLE', 'leaf_infLE']

    # create output dir.
    out_dir = os.path.join(args.out_dir,
                           args.exp,
                           f'exp_{exp_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    if args.exp == 'noise_set':
        get_noise_set_status(args, logger, out_dir, exp_hash)

    else:
        get_result_status(args, logger, out_dir, exp_hash)

    # clean up
    util.remove_logger(logger)

if __name__ == '__main__':
    main(status_args.get_experiments_args().parse_args())
