"""
Common commandline arguments.
"""
import configargparse


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
    cmd.add('--method', type=str, default='random')
    cmd.add('--leaf_inf_update_set', type=int, default=-1)  # LeafInfluence
    cmd.add('--leaf_inf_atol', type=int, default=1e-5)  # LeafInfluence
    cmd.add('--input_sim_measure', type=str, default='euclidean')  # InputSim
    cmd.add('--tree_sim_measure', type=str, default='dot_prod')  # TreeSim
    cmd.add('--tree_kernel', type=str, default='lpw')  # Trex, TreeSim
    cmd.add('--trex_target', type=str, default='actual')  # Trex
    cmd.add('--trex_lmbd', type=float, default=0.003)  # Trex
    cmd.add('--trex_n_epoch', type=str, default=3000)  # Trex
    cmd.add('--dshap_trunc_frac', type=float, default=0.25)  # DShap
    cmd.add('--dshap_check_every', type=int, default=100)  # DShap
    cmd.add('--subsample_sub_frac', type=float, default=0.7)  # SubSample
    cmd.add('--subsample_n_iter', type=int, default=4000)  # SubSample
    cmd.add('--n_jobs', type=int, default=-1)  # LOO, DShap, SubSample, LeafInf, LeafRefit
    cmd.add('--random_state', type=int, default=1)  # DShap, LOO, Minority, Random, SubSample, Target, Trex
    return cmd


# Single test example experiments


def get_influence_args():
    """
    Add arguments specific to the "Influence" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/influence/')
    cmd.add('--n_test', type=int, default=100)
    return cmd


def get_influenceLE_args():
    """
    Add arguments specific to the "InfluenceLE" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/influenceLE/')
    cmd.add('--n_test', type=int, default=100)
    return cmd


def get_remove_args():
    """
    Add arguments specific to the "Remove" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/remove/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    return cmd


def get_label_args():
    """
    Add arguments specific to the "Label" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/label/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    return cmd


def get_poison_args():
    """
    Add arguments specific to the "Poison" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/poison/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--poison_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    return cmd


def get_counterfactual_args():
    """
    Add arguments specific to the "Counterfactual" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/counterfactual/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_resources_args():
    """
    Add arguments specific to the "Resources" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/resources/')
    cmd.add('--n_repeat', type=int, default=5)
    cmd.add('--seed', type=int, default=-1)
    return cmd


def get_structure_args():
    """
    Add arguments specific to the "Structure" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/structure/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--n_remove', type=float, nargs='+', default=[1, 10, 100])
    return cmd


def get_reinfluence_args():
    """
    Add arguments specific to the "Reinfluence" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/reinfluence/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, default=0.02)
    cmd.add('--strategy', type=str, default='reestimate')
    cmd.add('--n_early_stop', type=int, default=0)
    return cmd


def get_label_edit_args():
    """
    Add arguments specific to the "Label Edit" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/label_edit/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--step_size', type=int, default=10)
    return cmd


def get_targeted_edit_args():
    """
    Add arguments specific to the "Targeted Edit" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence/')
    cmd.add('--out_dir', type=str, default='output/targeted_edit/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
    return cmd


# Set of test examples experiments


def get_influence_set_args():
    """
    Add arguments specific to the "Influence Set" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/influence_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    return cmd


def get_remove_set_args():
    """
    Add arguments specific to the "Remove Set" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence_set/')
    cmd.add('--out_dir', type=str, default='output/remove_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--remove_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    return cmd


def get_label_set_args():
    """
    Add arguments specific to the "Label Set" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence_set/')
    cmd.add('--out_dir', type=str, default='output/label_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--edit_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    return cmd


def get_poison_set_args():
    """
    Add arguments specific to the "Poison Set" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='output/influence_set/')
    cmd.add('--out_dir', type=str, default='output/poison_set/')
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--poison_frac', type=float, nargs='+',
            default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    return cmd


def get_noise_set_args():
    """
    Add arguments specific to the "Noise Set" experiment.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--out_dir', type=str, default='output/noise_set/')
    cmd.add('--strategy', type=str, default='test_sum')
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    return cmd
