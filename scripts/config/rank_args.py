"""
Rank commandline arguments.
"""
import configargparse

from . import post_args


def get_general_args(cmd=None):
    """
    Create an ArgParser object and add general arguments to it.

    Return ArgParser object.
    """
    if cmd is None:
        cmd = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    cmd.add('--data_dir', type=str, default='data/')
    cmd.add('--dataset_list', type=str, nargs='+',
            default=['adult', 'bank_marketing', 'bean', 'compas', 'concrete', 'credit_card',
                     'diabetes', 'energy', 'flight_delays', 'german_credit', 'htru2', 'life',
                     'naval', 'no_show', 'obesity', 'power', 'protein', 'spambase',
                     'surgical', 'twitter', 'vaccine', 'wine'])
    cmd.add('--tree_type', type=str, nargs='+', default=['lgb', 'xgb', 'sgb'])
    return cmd


def get_roar_args():
    """
    Add arguments specific to "Roar" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/roar/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd


def get_counterfactual_args():
    """
    Add arguments specific to the "Counterfactual" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/counterfactual/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_noise_args():
    """
    Add arguments specific to the "Noise" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/noise/')
    cmd.add('--noise_frac', type=float, default=0.4)  # TODO: average over multiple fracs.?
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd


def get_poison_args():
    """
    Add arguments specific to the "Poison" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/poison/')
    cmd.add('--poison_frac', type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3])
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd
