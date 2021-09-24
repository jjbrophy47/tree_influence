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


def get_roar_args():
    """
    Add arguments specific to "Roar" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_influence/')
    cmd.add('--out_dir', type=str, default='output/plot/roar/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--ckpt', type=int, default=1)
    return cmd


def get_counterfactual_args():
    """
    Add arguments specific to the "Counterfactual" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_counterfactual/')
    cmd.add('--out_dir', type=str, default='output/plot/counterfactual/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_correlation_args():
    """
    Add arguments specific to the "Correlation" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = exp_args.get_explainer_args(cmd)
    cmd.add('--method_list', type=str, nargs='+',
            default=['target', 'leaf_sim', 'boostin', 'trex', 'leaf_infSP', 'loo', 'subsample'])
    cmd.add('--skip', type=str, nargs='+', default=[])
    cmd.add('--in_dir', type=str, default='output/plot/correlation/')
    cmd.add('--out_dir', type=str, default='output/plot/correlation/')
    cmd.add('--sub_dir', type=str, default=None)
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, default=0.02)
    cmd.add('--n_ckpt', type=int, default=20)
    return cmd


def get_noise_args():
    """
    Add arguments specific to the "Noise" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_noise/')
    cmd.add('--out_dir', type=str, default='output/plot/noise/')
    cmd.add('--strategy', type=str, nargs='+', default=['self', 'test_sum'])
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3])
    cmd.add('--n_repeat', type=int, default=5)
    cmd.add('--ckpt', type=int, default=3)
    return cmd


def get_poison_args():
    """
    Add arguments specific to the "Poison" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_poison')
    cmd.add('--out_dir', type=str, default='output/plot/poison/')
    cmd.add('--poison_frac', type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3])
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--ckpt', type=int, default=3)
    return cmd


def get_resources_args():
    """
    Add arguments specific to the "Resources" summarization.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = post_args.get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_resources/')
    cmd.add('--out_dir', type=str, default='output/plot/resources/')
    cmd.add('--n_repeat', type=int, default=5)
    return cmd
