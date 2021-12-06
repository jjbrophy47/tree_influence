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
    cmd.add('--tree_type', type=str, nargs='+', default=['lgb', 'sgb', 'xgb', 'cb'])
    return cmd


# single test


def get_remove_args():
    """
    Add arguments specific to "Remove" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/remove/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd


def get_targeted_edit_args():
    """
    Add arguments specific to the "Targeted Edit" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/targeted_edit/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return cmd


# multi test


def get_remove_set_args():
    """
    Add arguments specific to the "Remove Set" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/remove_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--remove_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd


def get_label_set_args():
    """
    Add arguments specific to the "Label Set" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/label_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd


def get_noise_set_args():
    """
    Add arguments specific to the "Noise Set" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/noise_set/')
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    return cmd


# deprecated


def get_label_args():
    """
    Add arguments specific to "Label" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/label/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd


def get_poison_args():
    """
    Add arguments specific to "Poison" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/poison/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--poison_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
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


def get_removal_set_args():
    """
    Add arguments specific to the "Removal Set" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/removal_set/')
    cmd.add('--remove_frac', type=float,
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return cmd


def get_poison_set_args():
    """
    Add arguments specific to the "Poison Set" rank.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/plot/poison_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--poison_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return cmd
