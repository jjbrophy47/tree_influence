"""
Postprocessing commandline arguments.
"""
import configargparse

from . import exp_args


def get_general_args(cmd=None):
    """
    Create an ArgParser object and add general arguments to it.

    Return ArgParser object.
    """
    if cmd is None:
        cmd = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    cmd.add('--data_dir', type=str, default='data/')
    cmd.add('--dataset', type=str, default='surgical')
    cmd.add('--tree_type', type=str, default='lgb')
    return cmd


def get_explainer_args(cmd=None):
    """
    Add arguments used by the explainers.

    Input
        cmd: ArgParser, object to add commandline arguments to.

    Return ArgParser object.
    """
    if cmd is None:
        cmd = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    cmd.add('--method', type=str, nargs='+', default=['random', 'leaf_sim', 'boostin', 'leaf_infSP',
                                                      'trex', 'subsample', 'loo', 'leaf_inf', 'leaf_refit',
                                                      'target'])
    cmd.add('--skip', type=str, nargs='+', default=[])
    cmd.add('--leaf_inf_update_set', type=int, default=[-1])  # LeafInfluence
    cmd.add('--leaf_inf_atol', type=int, default=[1e-5])  # LeafInfluence
    cmd.add('--input_sim_measure', type=str, default=['euclidean'])  # InputSim
    cmd.add('--tree_sim_measure', type=str, default=['dot_prod'])  # TreeSim
    cmd.add('--tree_kernel', type=str, default=['lpw'])  # Trex, TreeSim
    cmd.add('--trex_target', type=str, default=['actual'])  # Trex
    cmd.add('--trex_lmbd', type=float, default=[0.003])  # Trex
    cmd.add('--trex_n_epoch', type=str, default=[3000])  # Trex
    cmd.add('--dshap_trunc_frac', type=float, default=[0.25])  # DShap
    cmd.add('--dshap_check_every', type=int, default=[100])  # DShap
    cmd.add('--subsample_sub_frac', type=float, default=[0.7])  # SubSample
    cmd.add('--subsample_n_iter', type=int, default=[4000])  # SubSample
    cmd.add('--n_jobs', type=int, default=-1)  # SubSample
    cmd.add('--random_state', type=int, default=1)  # SubSample
    return cmd


# Single test example


def get_remove_args():
    """
    Add arguments specific to "Remove" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_remove/')
    cmd.add('--out_dir', type=str, default='output/plot/remove/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--std_err', type=int, default=0)
    cmd.add('--custom_dir', type=str, default='')
    cmd.add('--legend', action='store_true')
    return cmd


def get_label_args():
    """
    Add arguments specific to "Label" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_label/')
    cmd.add('--out_dir', type=str, default='output/plot/label/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--std_err', type=int, default=0)
    cmd.add('--custom_dir', type=str, default='')
    return cmd


def get_poison_args():
    """
    Add arguments specific to "Label" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_poison/')
    cmd.add('--out_dir', type=str, default='output/plot/poison/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--poison_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--std_err', type=int, default=0)
    cmd.add('--custom_dir', type=str, default='')
    return cmd


def get_counterfactual_args():
    """
    Add arguments specific to the "Counterfactual" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_counterfactual/')
    cmd.add('--out_dir', type=str, default='output/plot/counterfactual/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_correlation_args():
    """
    Add arguments specific to the "Correlation" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = exp_args.get_explainer_args(cmd)
    cmd.add('--method_list', type=str, nargs='+',
            default=['random', 'target', 'input_sim', 'leaf_sim', 'boostin',
                     'trex', 'leaf_infSP', 'loo', 'subsample'])
    cmd.add('--skip', type=str, nargs='+', default=[])
    cmd.add('--in_dir', type=str, default='results/temp_influence/')
    cmd.add('--out_dir', type=str, default='output/plot/correlation/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--custom_dir', type=str, default=None)
    return cmd


def get_resources_args():
    """
    Add arguments specific to the "Resources" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_resources/')
    cmd.add('--out_dir', type=str, default='output/plot/resources/')
    cmd.add('--n_repeat', type=int, default=5)
    return cmd


def get_structure_args():
    """
    Add arguments specific to the "Structure" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_structure/')
    cmd.add('--out_dir', type=str, default='output/plot/structure/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--n_remove', type=float, nargs='+', default=[1, 10, 100])
    return cmd


def get_reinfluence_args():
    """
    Add arguments specific to "Reinfluence" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_reinfluence/')
    cmd.add('--out_dir', type=str, default='output/plot/reinfluence/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, default=0.02)
    cmd.add('--strategy', type=str, nargs='+', default=['fixed', 'reestimate'])
    cmd.add('--std_err', type=int, default=0)
    cmd.add('--ylabel', type=int, default=1)
    cmd.add('--legend1', type=int, default=1)
    cmd.add('--legend2', type=int, default=1)
    cmd.add('--markevery', type=int, default=1)
    return cmd


def get_label_edit_args():
    """
    Add arguments specific to the "Label Edit" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_label_edit/')
    cmd.add('--out_dir', type=str, default='output/plot/label_edit/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_targeted_edit_args():
    """
    Add arguments specific to the "Targeted Edit" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--method_list', type=str, nargs='+',
            default=['random', 'target', 'leaf_sim', 'trex', 'leaf_refit', 'leaf_inf', 'leaf_infSP',
                     'loo', 'subsample', 'boostin', 'boostinW1', 'boostinW2',
                     'boostinLE', 'boostinLEW1', 'boostinLEW2', 'looLE', 'leaf_refitLE',
                     'leaf_infLE', 'leaf_infSPLE'])
    cmd.add('--in_dir', type=str, default='results/temp_targeted_edit/')
    cmd.add('--out_dir', type=str, default='output/plot/targeted_edit/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    return cmd


# Set of test examples


def get_remove_set_args():
    """
    Add arguments specific to the "Remove Set" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_remove_set/')
    cmd.add('--out_dir', type=str, default='output/plot/remove_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--remove_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    return cmd


def get_label_set_args():
    """
    Add arguments specific to the "Label Set" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_label_set/')
    cmd.add('--out_dir', type=str, default='output/plot/label_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    return cmd


def get_poison_set_args():
    """
    Add arguments specific to the "Poison Set" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_poison_set/')
    cmd.add('--out_dir', type=str, default='output/plot/poison_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--poison_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    return cmd


def get_noise_set_args():
    """
    Add arguments specific to the "Noise Set" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_noise_set/')
    cmd.add('--out_dir', type=str, default='output/plot/noise_set/')
    cmd.add('--strategy', type=str, nargs='+', default=['self', 'test_sum'])
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    return cmd


# deprecated

def get_removal_set_args():
    """
    Add arguments specific to the "Removal Set" postprsocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='results/temp_removal_set/')
    cmd.add('--out_dir', type=str, default='output/plot/removal_set/')
    cmd.add('--remove_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    cmd.add('--val_frac', type=float, default=0.1)
    return cmd
