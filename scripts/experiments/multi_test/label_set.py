"""
Target the most helpful train examples and poison them.
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
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # config
sys.path.insert(0, here + '/../')  # util
import util
from config import exp_args
from single_test.influence import get_special_case_tol
from poison_set import poison


def experiment(args, logger, in_dir, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # get model
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)

    # read influence values and held-out test set
    inf_res = np.load(os.path.join(in_dir, 'results.npy'), allow_pickle=True)[()]
    influence = inf_res['influence']  # shape=(no. train,)
    val_idxs = inf_res['val_idxs']  # shape=(no. test,)
    test_idxs = inf_res['test_idxs']  # shape=(no. test,)

    X_test = X_test[test_idxs].copy()
    y_test = y_test[test_idxs].copy()

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. val.: {val_idxs.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # compute ranking
    ranking = np.argsort(influence)[::-1]  # most to least helpful

    # relabel most helpful train examples
    loss = []
    acc = []
    auc = []

    for edit_frac in args.edit_frac:
        n_edit = int(len(X_train) * edit_frac)
        edit_idxs = ranking[:n_edit]
        new_X_train, new_y_train = poison(X_train, y_train, objective, rng, edit_idxs, poison_features=False)

        # retrain and re-evaluate model on poisoned train data
        new_tree = clone(tree).fit(new_X_train, new_y_train)
        res_edit = util.eval_pred(objective, new_tree, X_test, y_test, logger,
                                  prefix=f'Test ({edit_frac * 100:>2.0f}% edit)')

        loss.append(res_edit['loss'])
        acc.append(res_edit['acc'])
        auc.append(res_edit['auc'])

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['edit_frac'] = np.array(args.edit_frac, dtype=np.float32)
    result['loss'] = np.array(loss, dtype=np.float32)
    result['acc'] = np.array(acc, dtype=np.float32)
    result['auc'] = np.array(auc, dtype=np.float32)
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for the explainer
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)
    _, method_hash = util.explainer_params_to_dict(args.method, vars(args))

    # get input dir., get unique hash for the influence experiment setting
    exp_dict = {'val_frac': args.val_frac}
    exp_hash = util.dict_to_hash(exp_dict)

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

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_label_set_args().parse_args())
