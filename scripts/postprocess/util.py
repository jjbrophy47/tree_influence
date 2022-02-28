import os
import sys
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util


# public
def get_results(args, exp_dir, logger=None, progress_bar=True):
    """
    Retrieve results for the multiple methods.
    """

    if logger and progress_bar:
        logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.method, args.leaf_inf_update_set, args.input_sim_measure,
                                         args.tree_sim_measure, args.tree_kernel, args.trex_target,
                                         args.trex_lmbd, args.trex_n_epoch,
                                         args.dshap_trunc_frac, args.dshap_check_every,
                                         args.subsample_sub_frac, args.subsample_n_iter]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings, disable=not progress_bar):

        method, leaf_inf_update_set, input_sim_measure, tree_sim_measure, tree_kernel, trex_target, trex_lmbd,\
            trex_n_epoch, dshap_trunc_frac, dshap_check_every, subsample_sub_frac, subsample_n_iter = items

        template = {'method': method,
                    'leaf_inf_update_set': leaf_inf_update_set,
                    'input_sim_measure': input_sim_measure,
                    'tree_sim_measure': tree_sim_measure,
                    'tree_kernel': tree_kernel,
                    'trex_target': trex_target,
                    'trex_lmbd': trex_lmbd,
                    'trex_n_epoch': trex_n_epoch,
                    'dshap_trunc_frac': dshap_trunc_frac,
                    'dshap_check_every': dshap_check_every,
                    'subsample_sub_frac': subsample_sub_frac,
                    'subsample_n_iter': subsample_n_iter,
                    'leaf_inf_atol': 1e-5,
                    'n_jobs': 1,
                    'random_state': 1}

        _, method_hash = exp_util.explainer_params_to_dict(method, template)
        method_dir = os.path.join(exp_dir, f'{method}_{method_hash}')
        method_id = f'{method}_{method_hash}'

        # skip empty experiments
        if not os.path.exists(method_dir) or method_id in visited:
            continue

        # add results to result dict
        else:
            visited.add(method_id)

            result = _get_result(template, method_dir)
            if result is not None:
                results.append((method_id, result))

    return results


def filter_results(results, skip_list):
    """
    Remove results for methods on the skip list.
    """

    result = []

    for method, res in results:

        include = True
        for skip in skip_list:

            if skip in method:
                include = False
                break

        if include:
            result.append((method, res))

    return result


def get_plot_dicts(markers=False):
    """
    Return dict for color, line, and labels for each method.
    """
    color = {}
    color['random_'] = 'blue'
    color['target_'] = 'cyan'
    color['minority_'] = 'cyan'
    color['loss_'] = 'yellow'
    color['boostin_'] = 'orange'
    color['boostinLE_'] = 'gold'
    color['boostinW1_'] = 'orange'
    color['boostinW1LE_'] = 'gold'
    color['boostinW2_'] = 'orange'
    color['boostinW2LE_'] = 'gold'
    color['leaf_infSP_'] = 'brown'
    color['leaf_infSPLE_'] = 'brown'
    color['leaf_sim_'] = 'gray'
    color['loo_'] = 'red'
    color['looLE_'] = 'red'
    color['trex_248fba74c1d2d9c0f9d547bb84083c21'] = 'green'
    color['input_sim_1edfa114070a90bb762993ab47712d68'] = 'gray'
    color['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'magenta'
    color['subsample_2b793a1ebcb67340112cf064fbf171cf'] = 'magenta'
    color['leaf_inf_6bb61e3b7bce0931da574d19d1d82c88'] = 'brown'
    color['leaf_infLE_6bb61e3b7bce0931da574d19d1d82c88'] = 'brown'
    color['leaf_refit_6bb61e3b7bce0931da574d19d1d82c88'] = 'purple'
    color['leaf_refitLE_6bb61e3b7bce0931da574d19d1d82c88'] = 'purple'

    line = {}
    line['random_'] = '-.'
    line['target_'] = '-.'
    line['minority_'] = '-'
    line['loss_'] = '-'
    line['boostin_'] = '-'
    line['boostinLE_'] = '-'
    line['boostinW1_'] = '--'
    line['boostinW1LE_'] = '--'
    line['boostinW2_'] = '-.'
    line['boostinW2LE_'] = '-.'
    line['leaf_infSP_'] = '-'
    line['leaf_infSPLE_'] = '--'
    line['leaf_sim_'] = '-'
    line['loo_'] = '--'
    line['looLE_'] = '--'
    line['trex_248fba74c1d2d9c0f9d547bb84083c21'] = '-'
    line['input_sim_1edfa114070a90bb762993ab47712d68'] = '--'
    line['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = '--'
    line['subsample_2b793a1ebcb67340112cf064fbf171cf'] = '--'
    line['leaf_inf_6bb61e3b7bce0931da574d19d1d82c88'] = '--'
    line['leaf_infLE_6bb61e3b7bce0931da574d19d1d82c88'] = '--'
    line['leaf_refit_6bb61e3b7bce0931da574d19d1d82c88'] = '--'
    line['leaf_refitLE_6bb61e3b7bce0931da574d19d1d82c88'] = '--'

    label = {}
    label['random_'] = 'Random'
    label['target_'] = 'RandomSL'
    label['minority_'] = 'Minority'
    label['loss_'] = 'Loss'
    label['boostin_'] = 'BoostIn'
    label['boostinLE_'] = 'BoostInLE'
    label['boostinW1_'] = 'BoostInW1'
    label['boostinW1LE_'] = 'BoostInW1LE'
    label['boostinW2_'] = 'BoostInW2'
    label['boostinW2LE_'] = 'BoostInW2LE'
    label['leaf_infSP_'] = 'LeafInfSP'
    label['leaf_infSPLE_'] = 'LeafInfSPLE'
    label['leaf_sim_'] = 'TreeSim'
    label['loo_'] = 'LOO'
    label['looLE_'] = 'LOOLE'
    label['trex_248fba74c1d2d9c0f9d547bb84083c21'] = 'TREX'
    label['input_sim_1edfa114070a90bb762993ab47712d68'] = 'Input Sim.'
    label['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'Data Shap'
    label['subsample_2b793a1ebcb67340112cf064fbf171cf'] = 'SubSample'
    label['leaf_inf_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafInfluence'
    label['leaf_infLE_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafInfluenceLE'
    label['leaf_refit_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafRefit'
    label['leaf_refitLE_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafRefitLE'

    marker = {}
    marker['random_'] = 'o'
    marker['target_'] = '*'
    marker['minority_'] = '.'
    marker['loss_'] = '.'
    marker['boostin_'] = 'o'
    marker['boostinLE_'] = '.'
    marker['boostinW1_'] = '.'
    marker['boostinW1LE_'] = '.'
    marker['boostinW2_'] = '.'
    marker['boostinW2LE_'] = '.'
    marker['leaf_infSP_'] = '*'
    marker['leaf_infSPLE_'] = '.'
    marker['leaf_sim_'] = 'd'
    marker['loo_'] = 'd'
    marker['looLE_'] = '.'
    marker['trex_248fba74c1d2d9c0f9d547bb84083c21'] = '+'
    marker['input_sim_1edfa114070a90bb762993ab47712d68'] = '.'
    marker['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = '.'
    marker['subsample_2b793a1ebcb67340112cf064fbf171cf'] = '+'
    marker['leaf_inf_6bb61e3b7bce0931da574d19d1d82c88'] = '*'
    marker['leaf_infLE_6bb61e3b7bce0931da574d19d1d82c88'] = '.'
    marker['leaf_refit_6bb61e3b7bce0931da574d19d1d82c88'] = 'o'
    marker['leaf_refitLE_6bb61e3b7bce0931da574d19d1d82c88'] = '.'

    result = (color, line, label)

    if markers:
        result += (marker,)

    return result


def plot_settings(family='serif', fontsize=11,
                  markersize=5, linewidth=None):
    """
    Matplotlib settings.
    """
    plt.rc('font', family=family)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('axes', titlesize=fontsize)
    plt.rc('legend', fontsize=fontsize)
    plt.rc('legend', title_fontsize=fontsize)
    plt.rc('lines', markersize=markersize)
    if linewidth is not None:
        plt.rc('lines', linewidth=linewidth)


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height


def get_method_color(method):
    """
    Return color given the method name.
    """
    color = {}
    color['Random'] = 'blue'
    color['Target'] = 'cyan'
    color['Minority'] = 'cyan'
    color['Loss'] = 'yellow'
    color['BoostIn'] = 'orange'
    color['LeafInfSP'] = 'brown'
    color['TREX'] = 'green'
    color['TreeSim'] = 'mediumseagreen'
    color['InputSim'] = 'gray'
    color['LOO'] = 'red'
    color['SubSample'] = 'rebeccapurple'
    color['LeafInfluence'] = 'brown'
    color['LeafRefit'] = 'gray'

    assert method in color, f'{method} not in color dict'
    return color[method]


# private
def _get_result(template, in_dir):
    """
    Obtain the results for this baseline method.
    """
    result = template.copy()

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]
        result.update(d)

    return result
