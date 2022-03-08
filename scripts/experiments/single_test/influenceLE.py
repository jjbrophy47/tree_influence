"""
Compute local influence.
"""
import os
import sys
import time
import joblib
import argparse
import resource
from datetime import datetime

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../../')  # tree_influence
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import tree_influence
import util
from config import exp_args
from influence import select_elements
from influence import get_special_case_tol


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


def get_adversarial_labels(tree, X_test, y_test, test_idxs, objective, y_train):
    """
    Return adversarial target labels and whether the test prediction is correct.

    Input
        tree: tree-ensemble model.
        X_test: 2d array of data.
        test_idxs: 1d array of test indices.
        objective: str, dataset task.

    Return 1d array of adversarial target labels, and a 1d array of boolea values.
    """
    y_train_median = np.median(y_train)

    is_correct_list = []
    adv_label_list = []

    for test_idx in test_idxs:
        X_test_temp = X_test[[test_idx]]
        y_test_temp = y_test[[test_idx]]

        res = util.eval_loss(objective, tree, X_test_temp, y_test_temp, logger=None, prefix=f'{0:>5}: {0:>5.2f}%')
        pred_lbl, pred_val, tgt_lbl, adv_lbl, is_correct = get_label(y_test_temp[0], res['pred'], objective,
                                                                     y_train_median)

        if objective == 'regression':
            if adv_lbl == 1:
                adv_lbl = y_train_median + (y_train_median / 2)
            else:
                adv_lbl = y_train_median - (y_train_median / 2)

        is_correct_list.append(is_correct)
        adv_label_list.append(adv_lbl)

    is_correct_arr = np.array(is_correct_list, dtype=bool)
    adv_labels_arr = np.array(adv_label_list, dtype=np.float32)

    return is_correct_arr, adv_labels_arr


def experiment(args, logger, params, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
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

    # randomly select test instances to compute influence values for
    avail_idxs = np.arange(X_test.shape[0])
    n_test = min(args.n_test, len(avail_idxs))
    test_idxs = select_elements(avail_idxs, rng, n=n_test)

    # get adversarial labels to flip training examples to
    is_correct_arr, adv_labels = get_adversarial_labels(tree, X_test, y_test, test_idxs, objective, y_train)

    # fit explainer
    start = time.time()
    explainer = tree_influence.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train,
                                                                              target_labels=adv_labels)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # compute influence
    start2 = time.time()

    # shape=(no. train, no. test)
    influence = explainer.get_local_influence(X_test[test_idxs], y_test[test_idxs], target_labels=adv_labels)
    inf_time = time.time() - start2

    logger.info(f'[INFO] explainer influence time: {inf_time:.5f}s')
    logger.info(f'[INFO] total time: {time.time() - begin:.5f}s')

    # save results
    result['influence'] = influence
    result['test_idxs'] = test_idxs
    result['is_correct_arr'] = is_correct_arr
    result['adv_labels'] = adv_labels
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = args.n_jobs

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    assert args.method.endswith('LE')

    # get unique hash for this experiment setting
    exp_dict = {'n_test': args.n_test}
    exp_hash = util.dict_to_hash(exp_dict)

    # get unique hash for the explainer
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)
    params, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # create output dir
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

    experiment(args, logger, params, out_dir)

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_influenceLE_args().parse_args())
