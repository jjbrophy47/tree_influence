"""
Identify and remove noisy examples.
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
from poison import poison
import intent
import util
from config import exp_args
from influence import get_special_case_tol


def add_noise(X, y, objective, rng, frac=0.1):
    """
    Add noise to a random subset of examples.
    """
    new_X = X
    new_y = y.copy()

    # select examples to add noise to
    target_idxs = rng.choice(np.arange(len(y)), size=int(len(y) * frac))

    # flip selected targets
    if objective == 'regression':
        new_y[target_idxs] = -y[target_idxs]

    # flip selected targets
    elif objective == 'binary':
        new_y[target_idxs] = np.where(y[target_idxs] == 0, 1, 0)

    # change selected targets to a different random label
    else:
        assert objective == 'multiclass'
        labels = np.unique(y)

        for target_idx in target_idxs:
            remain_labels = np.setdiff1d(labels, y[target_idx])
            new_y[target_idx] = rng.choice(remain_labels, size=1)

    return new_X, new_y, target_idxs


def experiment(args, logger, params, random_state, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(random_state)

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # add noise to a subset of the train examples
    X_train_noise, y_train_noise, noise_idxs = add_noise(X_train, y_train, objective, rng, frac=args.noise_frac)

    # use a fraction of the test data for validation
    stratify = None if objective == 'regression' else y_test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=1.0 - args.val_frac,
                                                    stratify=stratify, random_state=random_state)

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}, no. noisy: {len(noise_idxs):,}')
    logger.info(f'no. val.: {X_val.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train two ensembles: one with clean data and one with noisy data
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=random_state)
    tree.set_params(**hp)

    tree = tree.fit(X_train, y_train)
    res_clean = util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test (clean)')

    tree_noise = clone(tree).fit(X_train_noise, y_train_noise)
    res_noise = util.eval_pred(objective, tree_noise, X_test, y_test, logger, prefix='Test (noisy)')

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree_noise, X_train_noise, y_train_noise)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # compute influence
    start2 = time.time()

    if args.strategy == 'test_sum':
        local_influence = explainer.get_local_influence(X_val, y_val)  # shape=(no. test,)
        influence = np.sum(local_influence, axis=1)  # shape=(no. train,)
        ranking = np.argsort(influence)

    else:
        assert args.strategy == 'self'
        influence = explainer.get_self_influence(X_train_noise, y_train_noise)
        ranking = np.argsort(influence)[::-1]

    inf_time = time.time() - start2
    logger.info(f'\n[INFO] inf. time: {inf_time:.3f}s')

    # result containers
    result = {}
    result['check_frac'] = args.check_frac
    result['frac_detected'] = np.full(len(args.check_frac), np.nan, dtype=np.float32)
    result['loss'] = np.full(len(args.check_frac), np.nan, dtype=np.float32)
    result['acc'] = np.full(len(args.check_frac), np.nan, dtype=np.float32)
    result['auc'] = np.full(len(args.check_frac), np.nan, dtype=np.float32)

    for i, check_frac in enumerate(args.check_frac):

        # check and fix identified noisy examples
        n_check = int(len(X_train_noise) * check_frac)
        check_idxs = ranking[:n_check]

        new_X_train = X_train_noise.copy()
        new_X_train[check_idxs] = X_train[check_idxs]

        new_y_train = y_train_noise.copy()
        new_y_train[check_idxs] = y_train[check_idxs]

        logger.info(f'\n[INFO] checking {len(check_idxs):,} examples...')

        n_detected = len(set(noise_idxs).intersection(set(check_idxs)))
        frac_detected = n_detected / len(noise_idxs)
        logger.info(f'[INFO] no. noisy examples detected: '
                    f'{n_detected:,}/{len(noise_idxs):,} ({frac_detected * 100:.1f}%)')

        # retrain and re-evaluate model on fixed train data
        new_tree = clone(tree).fit(new_X_train, new_y_train)
        res_fixed = util.eval_pred(objective, new_tree, X_test, y_test, logger,
                                   prefix=f'Test (check/fixed {check_frac * 100:.1f}%)')

        # append results
        result['frac_detected'][i] = frac_detected
        result['loss'][i] = res_fixed['loss']
        result['acc'][i] = res_fixed['acc']
        result['auc'][i] = res_fixed['auc']

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['influence'] = influence
    result['noise_idxs'] = noise_idxs
    result['ranking'] = ranking
    result['res_clean'] = res_clean
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    for random_state in range(1, args.n_repeat + 1):

        # select seed
        if args.seed > 0:
            assert args.n_repeat == 1
            seed = args.seed

        else:
            seed = random_state

        # get unique hash for this experiment setting
        exp_dict = {'noise_frac': args.noise_frac, 'val_frac': args.val_frac,
                    'check_frac': args.check_frac}
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
                               args.strategy,
                               f'random_state_{seed}',
                               f'{args.method}_{method_hash}')

        # create output directory and clear previous contents
        os.makedirs(out_dir, exist_ok=True)
        util.clear_dir(out_dir)

        logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
        logger.info(args)
        logger.info(f'\ntimestamp: {datetime.now()}')

        experiment(args, logger, params, seed, out_dir)

        util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_noise_args().parse_args())
