"""
Compute change in loss as most influential examples
are edited with different labels.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import util
from config import exp_args
from influence import get_special_case_tol
from influence import select_n_jobs
from influence import select_elements
from influenceLE import get_label


def edit_labels(y, flip_idxs, objective, adv_label, y_median=None):
    """
    Change the labels in y for `flip_idxs` indices based on the `adv_label`.
    """
    new_y = y.copy()

    for flip_idx in flip_idxs:

        if objective == 'binary':
            new_y[flip_idx] = adv_label

        elif objective == 'regression':
            assert y_median is not None
            if adv_label == 0:
                new_y[flip_idx] = y_median - (y_median / 2)
            else:
                new_y[flip_idx] = y_median + (y_median / 2)

        else:
            assert objective == 'multiclass'
            new_y[flip_idx] = adv_label

    return new_y


def relabel_and_evaluate(test_idx, args, objective, tree,
                         X_train, y_train, X_test, y_test, influence,
                         is_le_method, is_correct_a, adv_lbl_a,
                         logger):

    # get appropriate evaluation function
    eval_fn = util.eval_loss

    res = eval_fn(objective, tree, X_test, y_test, logger=None, prefix=f'{0:>5}: {0:>5.2f}%')

    y_train_median = np.median(y_train)
    pred_lbl, pred_val, tgt_lbl, adv_lbl, is_correct = get_label(y_test[0], res['pred'], objective, y_train_median)

    logger.info(f'\n[Test ID {test_idx}], '
                f'pred.: {pred_val:.5f}, pred. label: {pred_lbl} '
                f'target: {y_test[0]:.5f}, target_label: {tgt_lbl}, '
                f'is_correct: {is_correct}, adv. target: {adv_lbl}, '
                f'loss: {res["loss"]:.3f}')

    # sanity checks
    if is_le_method:
        assert is_correct_a == is_correct

        if objective in ['binary', 'multiclass']:
            assert adv_lbl_a == adv_lbl

        else:
            assert objective == 'regression'
            if adv_lbl == 0:
                assert adv_lbl_a < y_train_median
            else:
                assert adv_lbl == 1
                assert adv_lbl_a > y_train_median

    ranking = np.argsort(influence)[::-1]  # shape=(no. train,)

    # result container
    result = {}
    result['loss'] = np.full(len(args.edit_frac), np.nan, dtype=np.float32)
    result['pred_label'] = np.full(len(args.edit_frac), np.nan, dtype=np.float32)

    for i, edit_frac in enumerate(args.edit_frac):
        n_edit = int(len(X_train) * edit_frac)

        edit_idxs = ranking[:n_edit]
        new_y_train = edit_labels(y_train, edit_idxs, objective, adv_lbl, y_train_median)

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            new_tree = clone(tree).fit(X_train, new_y_train)

            prefix = f'Labels flipped: {n_edit:>5} ({edit_frac * 100:>5.3f}%)'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)

            result['loss'][i] = res['loss']
            result['pred_label'][i] = res['pred_label']

    return result


def experiment(args, logger, in_dir, out_dir):

    # initialize experiment
    begin = time.time()
    result = {}
    rng = np.random.default_rng(args.random_state)

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

    # read influence values
    inf_res = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]
    influence = inf_res['influence']  # shape=(no. train, no. test)
    test_idxs = inf_res['test_idxs']  # shape=(no. test,)

    is_le_method = True if args.method.endswith('LE') else False
    if is_le_method:
        is_correct_list = inf_res['is_correct_arr']  # shape=(no. test,)
        adv_lbl_list = inf_res['adv_labels']  # shape=(no. test,)

    else:
        is_correct_list = [None] * influence.shape[1]
        adv_lbl_list = [None] * influence.shape[1]

    # get no. jobs to run in parallel
    n_jobs = select_n_jobs(args.n_jobs)
    logger.info(f'\n[INFO] no. jobs: {n_jobs:,}')

    with joblib.Parallel(n_jobs=n_jobs) as parallel:

        n_finish = 0
        n_remain = len(test_idxs)

        res_list = []

        while n_remain > 0:
            n = min(n_jobs, n_remain)

            results = parallel(joblib.delayed(relabel_and_evaluate)
                                             (idx, args, objective, tree,
                                              X_train, y_train, X_test[[idx]], y_test[[idx]],
                                              influence[:, n_finish + i], is_le_method, is_correct_list[n_finish + i],
                                              adv_lbl_list[n_finish + i],
                                              logger) for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

            # synchronization barrier
            res_list += results

            n_finish += n
            n_remain -= n

            cum_time = time.time() - begin
            logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                        f', cum. time: {cum_time:.3f}s')

    # save results
    result['edit_frac'] = np.array(args.edit_frac, dtype=np.float32)  # shape=(no. ckpts.,)
    result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['pred_label'] = np.vstack([res['pred_label'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for the explainer
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)
    params, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # get input dir., get unique hash for the influence experiment setting
    exp_dict = {'n_test': args.n_test}
    exp_hash = util.dict_to_hash(exp_dict)

    if args.method.endswith('LE'):
        args.in_dir = args.in_dir.replace('influence', 'influenceLE')

    in_dir = os.path.join(args.in_dir,
                          args.dataset,
                          args.tree_type,
                          f'exp_{exp_hash}',
                          f'{args.method}_{method_hash}')

    # create output dir., get unique hash for the influence experiment setting
    exp_dict['edit_frac'] = args.edit_frac
    exp_hash = util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{exp_hash}',
                           f'{args.method}_{method_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, in_dir, out_dir)


if __name__ == '__main__':
    main(exp_args.get_targeted_edit_args().parse_args())
