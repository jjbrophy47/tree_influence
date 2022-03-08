"""
Compute min. counterfactual influence set (editing examples).
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


def remove_and_evaluate(test_idx, objective, ranking, tree,
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
                f'is_correct: {is_correct}, adv. target: {adv_lbl}')

    # result container
    result = {}
    result['test_idx'] = test_idx
    result['start_pred'] = pred_val
    result['start_pred_label'] = pred_lbl

    if not is_correct:  # only try to flip correct predictions
        logger.info('Incorrect prediction, skipping...')
        result['status'] = 'skipped'
        result['status_code'] = -1
        return result

    for i in range(step_size, X_train.shape[0], step_size):

        flip_idxs = ranking[:i]
        new_y_train = edit_labels(y_train, flip_idxs, objective, adv_lbl, y_train_median)

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            result['status'] = 'fail'
            result['status_code'] = 2
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            result['status'] = 'fail'
            result['status_code'] = 2
            break

        else:
            edit_frac = i / X_train.shape[0]
            if edit_frac * 100 > 10.0:
                logger.info('Editing more than 10% of train, failed...')
                result['status'] = 'fail'
                result['status_code'] = 1
                break

            new_tree = clone(tree).fit(X_train, new_y_train)

            prefix = f'Labels flipped: {i:>5} ({edit_frac * 100:>5.3f}%)'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)

            new_lbl, new_pred_val, _, _, _ = get_label(y_test[0], res['pred'], objective, y_train_median)

            if new_lbl != pred_lbl:
                result['end_pred'] = new_pred_val
                result['end_pred_label'] = new_lbl
                result['n_edits'] = i
                result['frac_edits'] = float(edit_frac)
                result['status'] = 'success'
                result['status_code'] = 0
                break

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

            results = parallel(joblib.delayed(remove_and_evaluate)
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

    df = pd.DataFrame(res_list)

    # save results
    result['df'] = df
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
    _, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # get input dir., get unique hash for the influence experiment setting
    exp_dict = {'n_test': args.n_test}
    exp_hash = util.dict_to_hash(exp_dict)

    in_dir = os.path.join(args.in_dir,
                          args.dataset,
                          args.tree_type,
                          f'exp_{exp_hash}',
                          f'{args.method}_{method_hash}')

    # create output dir., get unique hash for the influence experiment setting
    exp_dict['step_size'] = args.step_size
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
    main(exp_args.get_counterfactual_args().parse_args())
