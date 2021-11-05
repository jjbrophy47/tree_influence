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
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import intent
import util
from config import exp_args
from influence import get_special_case_tol


def poison(X, y, objective, rng, target_idxs):
    """
    Add poison to training examples.
    """
    new_X = X.copy()
    new_y = y.copy()

    # replace feature values with mean values
    new_X[target_idxs] = np.mean(X, axis=0)

    # replace labels with random labels
    if objective in ['binary', 'multiclass']:
        new_y[target_idxs] = rng.choice(np.unique(y), size=len(target_idxs))

    return new_X, new_y


def experiment(args, logger, params, random_state, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # use a fraction of the test data for validation
    stratify = None if objective == 'regression' else y_test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=1.0 - args.val_frac,
                                                    stratify=stratify, random_state=random_state)

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. val.: {X_val.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train two ensembles: one with clean data and one with noisy data
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=random_state)
    tree.set_params(**hp)

    tree = tree.fit(X_train, y_train)
    res_clean = util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test (clean)')

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # compute influence
    start2 = time.time()
    local_influence = explainer.get_local_influence(X_val, y_val)  # shape=(no. test,)
    influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

    ranking = np.argsort(influence)[::-1]  # most to least helpful

    inf_time = time.time() - start2
    logger.info(f'\n[INFO] inf. time: {inf_time:.3f}s')

    # poison most helpful train examples
    loss = [res_clean['loss']]
    acc = [res_clean['acc']]
    auc = [res_clean['auc']]

    for poison_frac in args.poison_frac:
        n_poison = int(len(X_train) * poison_frac)
        poison_idxs = ranking[:n_poison]
        new_X_train, new_y_train = poison(X_train, y_train, objective, rng, poison_idxs)

        # retrain and re-evaluate model on poisoned train data
        new_tree = clone(tree).fit(new_X_train, new_y_train)
        res_poison = util.eval_pred(objective, new_tree, X_test, y_test, logger,
                                    prefix=f'Test ({poison_frac * 100:>2.0f}% poison)')

        loss.append(res_poison['loss'])
        acc.append(res_poison['acc'])
        auc.append(res_poison['auc'])

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['influence'] = influence
    result['poison_idxs'] = poison_idxs
    result['poison_frac'] = np.array([0.0] + args.poison_frac, dtype=np.float32)
    result['loss'] = np.array(loss, dtype=np.float32)
    result['acc'] = np.array(acc, dtype=np.float32)
    result['auc'] = np.array(auc, dtype=np.float32)
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()

    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))

    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'poison_frac': args.poison_frac, 'val_frac': args.val_frac}
    exp_hash = util.dict_to_hash(exp_dict)

    # special cases
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)

    # get unique hash for the explainer
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

    experiment(args, logger, params, args.random_state, out_dir)

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_poison_args().parse_args())
