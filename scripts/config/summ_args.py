"""
Summarization commandline arguments.
"""
import configargparse

from . import exp_args
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
    cmd.add('--tree_type', type=str, default='lgb')
    return cmd


# Single test example


def get_remove_args():
    """
    Add arguments specific to "Remove" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_remove/')
    cmd.add('--out_dir', type=str, default='output/plot/remove/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_targeted_edit_args():
    """
    Add arguments specific to the "Targeted Edit" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--method_list', type=str, nargs='+',
            default=['random', 'leaf_sim', 'trex',
                     'subsample', 'boostinLE', 'looLE',
                     'leaf_refitLE', 'leaf_infLE', 'leaf_infSPLE', 'target'])
    cmd.add('--in_dir', type=str, default='temp_targeted_edit/')
    cmd.add('--out_dir', type=str, default='output/plot/targeted_edit/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_correlation_args():
    """
    Add arguments specific to the "Correlation" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = exp_args.get_explainer_args(cmd)
    cmd.add('--method_list', type=str, nargs='+',
            default=['leaf_sim', 'boostin', 'trex', 'leaf_infSP', 'loo', 'subsample',
                     'leaf_inf', 'leaf_refit'])
    cmd.add('--skip', type=str, nargs='+', default=[])
    cmd.add('--in_dir', type=str, default='output/plot/correlation/')
    cmd.add('--in_sub_dir', type=str, default=None)
    cmd.add('--out_dir', type=str, default='output/plot/correlation/')
    cmd.add('--out_sub_dir', type=str, default=None)
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--tree_type_list', type=str, nargs='+', default=['lgb', 'sgb', 'xgb', 'cb'])
    return cmd


def get_resources_args():
    """
    Add arguments specific to the "Resources" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_resources/')
    cmd.add('--out_dir', type=str, default='output/plot/resources/')
    cmd.add('--n_repeat', type=int, default=5)
    cmd.add('--tree_type_list', type=str, nargs='+', default=['lgb', 'sgb', 'xgb', 'cb'])
    cmd.add('--use_existing', type=bool, default=True)
    return cmd


# Set of test examples


def get_remove_set_args():
    """
    Add arguments specific to the "Remove Set" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_remove_set')
    cmd.add('--out_dir', type=str, default='output/plot/remove_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--remove_frac', type=float,
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_label_set_args():
    """
    Add arguments specific to the "Label Set" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_label_set')
    cmd.add('--out_dir', type=str, default='output/plot/label_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--edit_frac', type=float,
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_noise_set_args():
    """
    Add arguments specific to the "Noise Set" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_noise_set/')
    cmd.add('--out_dir', type=str, default='output/plot/noise_set/')
    cmd.add('--strategy', type=str, nargs='+', default=['self', 'test_sum'])
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


# deprecated


def get_label_args():
    """
    Add arguments specific to "Label" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_label/')
    cmd.add('--out_dir', type=str, default='output/plot/label/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_poison_args():
    """
    Add arguments specific to "Poison" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_poison/')
    cmd.add('--out_dir', type=str, default='output/plot/poison/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--poison_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_counterfactual_args():
    """
    Add arguments specific to the "Counterfactual" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_counterfactual/')
    cmd.add('--out_dir', type=str, default='output/plot/counterfactual/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_removal_set_args():
    """
    Add arguments specific to the "Removal Set" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_removal_set/')
    cmd.add('--out_dir', type=str, default='output/plot/removal_set/')
    cmd.add('--remove_frac', type=float,
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_poison_set_args():
    """
    Add arguments specific to the "Poison Set" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_poison_set')
    cmd.add('--out_dir', type=str, default='output/plot/poison_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--poison_frac', type=float,
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--ckpt', type=int, default=1)
    return cmd
