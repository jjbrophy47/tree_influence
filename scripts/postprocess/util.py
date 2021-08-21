import os
import sys
from itertools import product

import numpy as np
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util


# public
def get_results(args, in_dir, logger=None):
    """
    Retrieve results for the multiple methods.
    """

    if logger:
        logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.method, args.leaf_scale, args.update_set,
                                         args.kernel, args.target, args.lmbd, args.n_epoch, args.trunc_frac,
                                         args.check_every, args.sub_frac, args.n_iter,
                                         args.global_op, args.local_op, args.similarity]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings):

        method, leaf_scale, update_set, kernel, target, lmbd, n_epoch,\
            trunc_frac, check_every, sub_frac, n_iter, global_op, local_op, similarity = items

        template = {'method': method,
                    'leaf_scale': leaf_scale,
                    'update_set': update_set,
                    'kernel': kernel,
                    'target': target,
                    'lmbd': lmbd,
                    'similarity': similarity,
                    'n_epoch': n_epoch,
                    'trunc_frac': trunc_frac,
                    'check_every': check_every,
                    'sub_frac': sub_frac,
                    'n_iter': n_iter,
                    'random_state': args.random_state,
                    'n_jobs': args.n_jobs,
                    'global_op': global_op,
                    'local_op': local_op,
                    'similarity': similarity}

        exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
                    'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
        exp_hash = exp_util.dict_to_hash(exp_dict)

        _, hash_str = exp_util.explainer_params_to_dict(method, template)

        exp_dir = os.path.join(in_dir,
                               args.dataset,
                               args.tree_type,
                               f'exp_{exp_hash}',
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
    color = {'random_': 'blue', 'minority_': 'pink', 'target_': 'cyan', 'loss_': 'yellow'}
    # color['boostin_5fa0dc231efe739492c9904ea2304460'] = 'orange'
    # color['boostin_814cd0d3199b076a65b67e6d3b017c5f'] = 'orange'
    # color['boostin_92588165d78cfa923e46d8af4451baf2'] = 'orange'
    # color['boostin_61ee96c878fb97f78f3f309129890cf2'] = 'gray'
    # color['boostin_0b960733ca2491723f810be9e3bb84bf'] = 'gray'
    # color['boostin_0c30d2bb70334ce6bfbba141733017cc'] = 'gray'
    color['boostin2_fea087517c26fadd409bd4b9dc642555'] = 'orange'
    color['boostin3_fea087517c26fadd409bd4b9dc642555'] = 'orange'
    color['boostin4_fea087517c26fadd409bd4b9dc642555'] = 'orange'
    color['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = 'green'
    color['trex_c026a1d65c79084fe50ec2a8524b2533'] = 'green'
    color['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = 'green'
    color['loo_590f53e8699817c6fa498cc11a4cbe63'] = 'red'
    color['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = 'red'
    color['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'magenta'
    color['subsample_2b793a1ebcb67340112cf064fbf171cf'] = 'magenta'
    color['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = 'brown'
    color['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = 'brown'
    color['leaf_influenceSP_fea087517c26fadd409bd4b9dc642555'] = 'brown'
    color['similarity_da2995ca8d4801840027a5128211b2d0'] = 'gray'

    line = {'random_': '-', 'minority_': '-', 'target_': '-', 'loss_': '-'}
    # line['boostin_5fa0dc231efe739492c9904ea2304460'] = '-'
    # line['boostin_814cd0d3199b076a65b67e6d3b017c5f'] = '--'
    # line['boostin_92588165d78cfa923e46d8af4451baf2'] = ':'
    # line['boostin_61ee96c878fb97f78f3f309129890cf2'] = '-'
    # line['boostin_0b960733ca2491723f810be9e3bb84bf'] = '--'
    # line['boostin_0c30d2bb70334ce6bfbba141733017cc'] = ':'
    line['boostin2_fea087517c26fadd409bd4b9dc642555'] = '-'
    line['boostin3_fea087517c26fadd409bd4b9dc642555'] = '--'
    line['boostin4_fea087517c26fadd409bd4b9dc642555'] = ':'
    line['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = '-'
    line['trex_c026a1d65c79084fe50ec2a8524b2533'] = '--'
    line['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = ':'
    line['loo_590f53e8699817c6fa498cc11a4cbe63'] = '-'
    line['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = '--'
    line['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = '-'
    line['subsample_2b793a1ebcb67340112cf064fbf171cf'] = '-'
    line['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = '-'
    line['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = '--'
    line['leaf_influenceSP_fea087517c26fadd409bd4b9dc642555'] = ':'
    line['similarity_da2995ca8d4801840027a5128211b2d0'] = '-'

    label = {'random_': 'Random', 'minority_': 'Minority', 'target_': 'Target', 'loss_': 'Loss'}
    # label['boostin_5fa0dc231efe739492c9904ea2304460'] = 'BoostIn_0'
    # label['boostin_814cd0d3199b076a65b67e6d3b017c5f'] = 'BoostIn_-1'
    # label['boostin_92588165d78cfa923e46d8af4451baf2'] = 'BoostIn_-2'
    # label['boostin_61ee96c878fb97f78f3f309129890cf2'] = 'BoostIn_sign_0'
    # label['boostin_0b960733ca2491723f810be9e3bb84bf'] = 'BoostIn_sign_-1'
    # label['boostin_0c30d2bb70334ce6bfbba141733017cc'] = 'BoostIn_sign_-2'
    label['boostin2_fea087517c26fadd409bd4b9dc642555'] = 'BoostIn2'
    label['boostin3_fea087517c26fadd409bd4b9dc642555'] = 'BoostIn3'
    label['boostin4_fea087517c26fadd409bd4b9dc642555'] = 'BoostIn4'
    label['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = 'TREX'
    label['trex_c026a1d65c79084fe50ec2a8524b2533'] = 'TREX_exp'
    label['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = 'TREX_alpha'
    label['loo_590f53e8699817c6fa498cc11a4cbe63'] = 'LOO'
    label['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = 'LOO_exp'
    label['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'DShap'
    label['subsample_2b793a1ebcb67340112cf064fbf171cf'] = 'SubSample'
    label['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafInf'
    label['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = 'LeafInf_SP'
    label['leaf_influenceSP_fea087517c26fadd409bd4b9dc642555'] = 'LeafInf_SP2'
    label['similarity_da2995ca8d4801840027a5128211b2d0'] = 'Sim.'

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
