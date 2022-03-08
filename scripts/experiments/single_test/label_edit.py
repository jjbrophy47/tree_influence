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
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import tree_influence
import util
from config import exp_args
from influence import get_special_case_tol
from structure import compute_affinity
from structure import compute_weighted_affinity


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


def relabel_and_evaluate(test_idx, objective, ranking, tree,
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

    computed_affinity = False

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

            # Audit: compute structural changes in the model
            if edit_frac * 100 >= 1.0 and not computed_affinity:
                logger.info('Reached 1% of train')

                logger.info('\tcomputing train affinity for initial model...')
                new_explainer = tree_influence.TreeExplainer('boostin', {}, logger).fit(tree, X_train, y_train)
                result['affinity'] = [compute_affinity(new_explainer.model_, X_train[flip_idxs], X_test)]
                result['affinity_edit_frac'] = [0]

                logger.info('\tcomputing weighted train affinity for initial model...')
                new_explainer.model_.update_node_count(X_train)
                result['weighted_affinity'] = [compute_weighted_affinity(new_explainer.model_,
                                                                         X_train[flip_idxs], X_test)]

                logger.info('\tcomputing train affinity for 1% edited model...')
                new_explainer = tree_influence.TreeExplainer('boostin', {}, logger).fit(new_tree, X_train, new_y_train)
                result['affinity'].append(compute_affinity(new_explainer.model_, X_train[flip_idxs], X_test))
                result['affinity_edit_frac'].append(edit_frac)

                logger.info('\tcomputing weighted train affinity for 1% edited model...')
                new_explainer.model_.update_node_count(X_train)
                result['weighted_affinity'].append(compute_weighted_affinity(new_explainer.model_,
                                                                             X_train[flip_idxs], X_test))

                computed_affinity = True

            prefix = f'Labels flipped: {i:>5} ({edit_frac * 100:>5.3f}%)'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)

            result['loss'].append(res['loss'])
            result['edit_frac'].append(edit_frac)

    result['loss'] = np.array(result['loss'], dtype=np.float32)
    result['edit_frac'] = np.array(result['edit_frac'], dtype=np.float32)

    return result


def experiment(args, logger, in_dir, out_dir):

    # initialize experiment
    begin = time.time()
    result = {}

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

    # get ranking
    ranking = np.argsort(influence, axis=0)[::-1]  # shape=(no. train, no. test)

    # get no. jobs to run in parallel
    if args.n_jobs == -1:
        n_jobs = joblib.cpu_count()

    else:
        assert args.n_jobs >= 1
        n_jobs = min(args.n_jobs, joblib.cpu_count())

    logger.info(f'\n[INFO] no. jobs: {n_jobs:,}')

    with joblib.Parallel(n_jobs=n_jobs) as parallel:

        n_finish = 0
        n_remain = len(test_idxs)

        res_list = []

        while n_remain > 0:
            n = min(n_jobs, n_remain)

            results = parallel(joblib.delayed(relabel_and_evaluate)
                                             (idx, objective, ranking[:, n_finish + i], tree,
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
    affinity_edit_frac = None

    for res in res_list:
        if res is not None:
            edit_frac = res['edit_frac']
            affinity_edit_frac = res['affinity_edit_frac']
            break

    assert edit_frac is not None
    assert affinity_edit_frac is not None

    # filter results
    res_list = [res for res in res_list if res is not None]

    # save results
    result['edit_frac'] = edit_frac  # shape=(no. ckpts,)
    result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['affinity'] = np.dstack([res['affinity'] for res in res_list])  # shape=(2, no. train, no. test)
    result['weighted_affinity'] = np.dstack([res['weighted_affinity'] for res in res_list])  # same as affinity
    result['affinity_edit_frac'] = affinity_edit_frac  # shape=(%tr,)
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
    _, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    exp_dict = {'n_test': args.n_test, 'remove_frac': args.remove_frac}

    # get input dir., get unique hash for the influence experiment setting
    in_exp_hash = util.dict_to_hash(exp_dict)

    exp_dir = os.path.join(args.in_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{in_exp_hash}',
                           f'{args.method}_{method_hash}')

    # create output dir., get unique hash for the influence experiment setting
    exp_dict['step_size'] = args.step_size
    out_exp_hash = util.dict_to_hash(exp_dict)

    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.tree_type,
                           f'exp_{out_exp_hash}',
                           f'{args.method}_{method_hash}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, exp_dir, out_dir)


if __name__ == '__main__':
    main(exp_args.get_label_edit_args().parse_args())
