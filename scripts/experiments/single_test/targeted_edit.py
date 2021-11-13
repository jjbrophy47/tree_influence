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
sys.path.insert(0, here + '/../../../')  # intent
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import intent
import util
from config import exp_args
from influence import get_special_case_tol
from influence import select_n_jobs
from influence import select_elements


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


def get_label(y, pred, objective, y_median=False):
    """
    Return predicted label, confidence, and adversarial target label,
        and bool whether or not the predicted label is correct.
    """

    if objective == 'binary':
        assert pred.ndim == 2

        target_label = y
        pred_label = np.argmax(pred)
        pred_val = pred[0][pred_label]
        adv_label = 1 if pred_label == 0 else 0
        is_correct = True if pred_label == y else False

    elif objective == 'regression':
        assert pred.ndim == 1
        assert y_median is not None

        target_label = 1 if y > y_median else 0
        pred_label = 1 if pred[0] > y_median else 0
        pred_val = pred[0]
        adv_label = 1 if pred_label == 0 else 0
        is_correct = True if pred_label == target_label else False

    else:
        assert objective == 'multiclass'
        assert pred.ndim == 1

        target_label = y
        sorted_labels = np.argsort(pred)[::-1]
        pred_label = sorted_labels[0]
        pred_val = pred[pred_label]
        adv_label = sorted_labels[1]  # 2nd most likely
        is_correct = True if pred_label == y else False

    return pred_label, pred_val, target_label, adv_label, is_correct


def relabel_and_evaluate(test_idx, args, params, objective, tree,
                         X_train, y_train, X_test, y_test,
                         remove_frac, step_size, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_loss

    res = eval_fn(objective, tree, X_test, y_test, None, prefix=f'{0:>5}: {0:>5.2f}%')

    y_train_median = np.median(y_train)
    pred_lbl, pred_val, tgt_lbl, adv_lbl, is_correct = get_label(y_test[0], res['pred'], objective, y_train_median)

    logger.info(f'\n[Test ID {test_idx}], '
                f'pred.: {pred_val:.5f}, pred. label: {pred_lbl} '
                f'target: {y_test[0]:.5f}, target_label {tgt_lbl}, '
                f'is_correct: {is_correct}, adv. target: {adv_lbl}, '
                f'loss: {res["loss"]:.3f}')

    # fit explainer
    if 'boostinLE' in args.method:

        if objective == 'regression':

            if adv_lbl == 1:
                fill_val = y_train_median - (y_train_median / 2)
            else:
                fill_val = y_train_median + (y_train_median / 2)

        else:
            fill_val = adv_lbl

        new_y_train = np.full(y_train.shape, fill_val, dtype=y_train.dtype)

    else:
        new_y_train = None

    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train, new_y=new_y_train)
    logger.info(f'\n[INFO] explainer fit time: {time.time() - start - explainer.parse_time_:.5f}s')

    # compute influence
    start2 = time.time()
    influence = explainer.get_local_influence(X_test, y_test)  # (no. train, 1)
    logger.info(f'\n[INFO] influence time: {time.time() - start:.5f}s')

    # get ranking
    ranking = np.argsort(influence[:, 0])[::-1]  # shape=(no. train,)

    # result container
    result = {}
    result['test_idx'] = test_idx
    result['start_pred'] = pred_val
    result['start_pred_label'] = pred_lbl
    result['loss'] = [res['loss']]
    result['edit_frac'] = [0]

    if not is_correct:  # only try to flip correct predictions
        logger.info('Incorrect prediction, skipping...')
        return None

    # computed_affinity = False

    for i in range(step_size, X_train.shape[0], step_size):

        flip_idxs = ranking[:i]
        new_y_train = edit_labels(y_train, flip_idxs, objective, adv_lbl, y_train_median)

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            edit_frac = i / X_train.shape[0]

            if edit_frac * 100 > 2.0:
                logger.info('Reached more than 2% of train, stopping...')
                break

            new_tree = clone(tree).fit(X_train, new_y_train)

            prefix = f'Labels flipped: {i:>5} ({edit_frac * 100:>5.3f}%)'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)

            result['loss'].append(res['loss'])
            result['edit_frac'].append(edit_frac)

    result['loss'] = np.array(result['loss'], dtype=np.float32)
    result['edit_frac'] = np.array(result['edit_frac'], dtype=np.float32)

    return result


def experiment(args, logger, out_dir, params):

    # initialize experiment
    begin = time.time()
    result = {}
    rng = np.random.default_rng(args.random_state)

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # randomly select test instances to compute influence values for
    avail_idxs = np.arange(X_test.shape[0])
    n_test = min(args.n_test, len(avail_idxs))
    test_idxs = select_elements(avail_idxs, rng, n=n_test)

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)
    tree = tree.fit(X_train, y_train)
    util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test')

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
                                             (idx, args, params, objective, tree,
                                              X_train, y_train, X_test[[idx]], y_test[[idx]],
                                              args.remove_frac, args.step_size, logger)
                                              for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

            # synchronization barrier
            res_list += results

            n_finish += n
            n_remain -= n

            cum_time = time.time() - begin
            logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                        f', cum. time: {cum_time:.3f}s')

    # get edit_frac array
    edit_frac = None
    # affinity_edit_frac = None

    for res in res_list:
        if res is not None:
            edit_frac = res['edit_frac']
            break

    assert edit_frac is not None

    # filter results
    res_list = [res for res in res_list if res is not None]

    # save results
    result['edit_frac'] = edit_frac  # shape=(no. ckpts,)
    result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # special cases
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)

    # get unique hash for the explainer
    params, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # create output dir., get unique hash for the influence experiment setting
    exp_dict = {'n_test': args.n_test, 'step_size': args.step_size}
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

    experiment(args, logger, out_dir, params)


if __name__ == '__main__':
    main(exp_args.get_targeted_edit_args().parse_args())
