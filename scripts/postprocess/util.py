import os
import sys
from itertools import product

import numpy as np
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util


# public
def get_results(args, logger=None):
    """
    Retrieve results for the multiple methods.
    """

    if logger:
        logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.method, args.use_leaf, args.update_set,
                                         args.kernel, args.target, args.lmbd, args.n_epoch,
                                         args.trunc_frac, args.check_every, args.global_op]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings):

        method, use_leaf, update_set, kernel, target, lmbd, n_epoch,\
            trunc_frac, check_every, global_op = items

        template = {'method': method,
                    'use_leaf': use_leaf,
                    'update_set': update_set,
                    'kernel': kernel,
                    'target': target,
                    'lmbd': lmbd,
                    'n_epoch': n_epoch,
                    'trunc_frac': trunc_frac,
                    'check_every': check_every,
                    'random_state': args.random_state,
                    'n_jobs': args.n_jobs,
                    'global_op': global_op}

        _, hash_str = exp_util.explainer_params_to_dict(method, template)

        exp_dir = os.path.join(args.in_dir,
                               args.dataset,
                               args.tree_type,
                               f'rs_{args.random_state}',
                               args.inf_obj,
                               f'{method}_{hash_str}')

        method_id = f'{method}_{hash_str}'

        # skip empty experiments
        if not os.path.exists(exp_dir) or method_id in visited:
            continue

        # add results to result dict
        else:
            visited.add(method_id)

            result = _get_result(template, exp_dir)
            if result is not None:
                results.append((method_id, result))

    return results


def get_plot_dicts():
    """
    Return dict for color, line, and labels for each method.
    """
    color = {'random_': 'blue', 'minority_': 'cyan'}
    color['boostin_c4ca4238a0b923820dcc509a6f75849b'] = 'orange'
    color['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = 'green'
    color['trex_c026a1d65c79084fe50ec2a8524b2533'] = 'green'
    color['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = 'green'
    color['loo_590f53e8699817c6fa498cc11a4cbe63'] = 'red'
    color['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = 'red'
    color['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'magenta'
    color['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = 'brown'
    color['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = 'brown'

    line = {'random_': '-', 'minority_': '-'}
    line['boostin_c4ca4238a0b923820dcc509a6f75849b'] = '-'
    line['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = '-'
    line['trex_c026a1d65c79084fe50ec2a8524b2533'] = '--'
    line['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = ':'
    line['loo_590f53e8699817c6fa498cc11a4cbe63'] = '-'
    line['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = '--'
    line['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = '-'
    line['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = '-'
    line['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = '--'

    label = {'random_': 'Random', 'minority_': 'Minority'}
    label['boostin_c4ca4238a0b923820dcc509a6f75849b'] = 'BoostIn'
    label['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = 'TREX'
    label['trex_c026a1d65c79084fe50ec2a8524b2533'] = 'TREX_exp'
    label['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = 'TREX_alpha'
    label['loo_590f53e8699817c6fa498cc11a4cbe63'] = 'LOO'
    label['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = 'LOO_exp'
    label['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'DShap'
    label['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafInf'
    label['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = 'LeafInf_SP'

    return color, line, label


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
