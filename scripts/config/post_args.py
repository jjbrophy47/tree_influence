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
    cmd.add('--method', type=str, nargs='+', default=['random', 'target', 'leaf_sim', 'boostin',
                                                      'leaf_infSP', 'trex', 'subsample', 'loo', 'leaf_inf',
                                                      'leaf_refit'])
    cmd.add('--skip', type=str, nargs='+', default=[])
    cmd.add('--leaf_inf_update_set', type=int, default=[-1])  # LeafInfluence
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


def get_roar_args():
    """
    Add arguments specific to "Roar" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_influence/')
    cmd.add('--out_dir', type=str, default='output/plot/roar/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
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
    cmd.add('--in_dir', type=str, default='temp_counterfactual/')
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
    cmd.add('--in_dir', type=str, default='temp_influence/')
    cmd.add('--out_dir', type=str, default='output/plot/correlation/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, nargs='+', default=[0.0, 0.001, 0.005, 0.01, 0.015, 0.02])
    cmd.add('--custom_dir', type=str, default=None)
    return cmd


def get_noise_args():
    """
    Add arguments specific to the "Noise" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_noise/')
    cmd.add('--out_dir', type=str, default='output/plot/noise/')
    cmd.add('--strategy', type=str, nargs='+', default=['self', 'test_sum'])
    cmd.add('--noise_frac', type=float, default=0.4)
    cmd.add('--val_frac', type=float, default=0.1)
    cmd.add('--check_frac', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3])
    cmd.add('--n_repeat', type=int, default=5)
    return cmd


def get_poison_args():
    """
    Add arguments specific to the "Poison" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_poison/')
    cmd.add('--out_dir', type=str, default='output/plot/poison/')
    cmd.add('--poison_frac', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2, 0.3])
    cmd.add('--val_frac', type=float, default=0.1)
    return cmd


def get_resources_args():
    """
    Add arguments specific to the "Resources" postprocessing.

    Return ArgParser object.
    """
    cmd = get_general_args()
    cmd = get_explainer_args(cmd)
    cmd.add('--in_dir', type=str, default='temp_resources/')
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
    cmd.add('--in_dir', type=str, default='temp_structure/')
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
    cmd.add('--in_dir', type=str, default='temp_reinfluence/')
    cmd.add('--out_dir', type=str, default='output/plot/reinfluence/')
    cmd.add('--n_test', type=int, default=100)
    cmd.add('--remove_frac', type=float, default=0.02)
    cmd.add('--strategy', type=str, nargs='+', default=['fixed', 'reestimate'])
    cmd.add('--std_err', type=int, default=0)
    cmd.add('--ylabel', type=int, default=1)
    cmd.add('--legend1', type=int, default=1)
    cmd.add('--legend2', type=int, default=1)
    return cmd
