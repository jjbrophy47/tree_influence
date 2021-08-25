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
import intent
import util


def jaccard_similarity(inf1, inf2):
    """
    Return |inf1 intersect inf2| / |inf1 union inf2|.
    """
    s1 = set(inf1)
    s2 = set(inf2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def add_target_noise(y, objective, rng, frac=0.1):
    """
    Add noise to ground-truth targets.
    """
    result = y.copy()

    # select exmaples to add noise to
    idxs = rng.choice(np.arange(len(y)), size=int(len(y) * frac))

    # add gaussian noise to selected targets
    if objective == 'regression':
        result[idxs] += rng.normal(np.median(y), np.std(y))

    # flip selected targets
    elif objective == 'binary':
        result[idxs] = np.where(y[idxs] == 0, 1, 0)

    # change selected targets to a different random label
    else:
        assert objective == 'multiclass'
        labels = np.unique(y)

        for idx in idxs:
            remain_labels = np.setdiff1d(labels, y[idx])
            result[idx] = rng.choice(remain_labels, size=1)

    return result, idxs


def add_feature_noise(X, objective):
    pass


def experiment(args, logger, params, random_state, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # add noise to a subset of the train examples
    if args.noise == 'target':
        X_train_noise = X_train
        y_train_noise, noise_idxs = add_target_noise(y_train, objective, rng, frac=args.noise_frac)

    else:
        assert args.noise == 'feature'
        y_train_noise = y_train
        X_train_noise, noise_idxs = add_feature_noise(X_train, objective, rng, frac=args.noise_frac)

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

    n_check = int(len(X_train_noise) * args.check_frac)
    check_idxs = ranking[:n_check]

    inf_time = time.time() - start2
    logger.info(f'\n[INFO] inf. time: {inf_time:.3f}s')

    # check and fix identified noisy examples
    new_X_train = X_train_noise.copy()
    new_X_train[check_idxs] = X_train[check_idxs]

    new_y_train = y_train_noise.copy()
    new_y_train[check_idxs] = y_train[check_idxs]

    logger.info(f'\n[INFO] checking {len(check_idxs):,} examples...')

    n_detected = len(set(noise_idxs).intersection(set(check_idxs)))
    frac_detected = n_detected / len(check_idxs)
    overall_frac_detected = n_detected / len(noise_idxs)
    logger.info(f'[INFO] no. noisy examples detected: '
                f'{n_detected:,}/{len(check_idxs):,} ({frac_detected * 100:.1f}%), '
                f'overall: {n_detected:,}/{len(noise_idxs):,} ({overall_frac_detected * 100:.1f}%)')

    # retrain and re-evaluate model on fixed train data
    new_tree = clone(tree).fit(new_X_train, new_y_train)
    res_fixed = util.eval_pred(objective, new_tree, X_test, y_test, logger, prefix='Test (fixed)')

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['influence'] = influence
    result['noise_idxs'] = noise_idxs
    result['check_idxs'] = check_idxs
    result['res_clean'] = res_clean
    result['res_noise'] = res_noise
    result['res_fixed'] = res_fixed
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

        # get unique hash for this experiment setting
        exp_dict = {'noise': args.noise, 'noise_frac': args.noise_frac,
                    'val_frac': args.val_frac, 'check_frac': args.check_frac}
        exp_hash = util.dict_to_hash(exp_dict)

        # get unique hash for the explainer
        params, hash_str = util.explainer_params_to_dict(args.method, vars(args))

        # special cases
        if args.method == 'leaf_influence':
            if args.dataset == 'flight_delays':
                params['atol'] = 1e-1

        # create output dir
        out_dir = os.path.join(args.out_dir,
                               args.dataset,
                               args.tree_type,
                               f'exp_{exp_hash}',
                               args.strategy,
                               f'random_state_{random_state}',
                               f'{args.method}_{hash_str}')

        # create output directory and clear previous contents
        os.makedirs(out_dir, exist_ok=True)
        util.clear_dir(out_dir)

        logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
        logger.info(args)
        logger.info(f'\ntimestamp: {datetime.now()}')

        experiment(args, logger, params, random_state, out_dir)

        util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/noise/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--strategy', type=str, default='test_sum')
    parser.add_argument('--noise', type=str, default='target')
    parser.add_argument('--noise_frac', type=float, default=0.2)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--check_frac', type=float, default=0.1)

    # Explainer settings
    parser.add_argument('--method', type=str, default='random')

    parser.add_argument('--leaf_scale', type=float, default=-1.0)  # BoostIn
    parser.add_argument('--local_op', type=str, default='normal')  # BoostIn

    parser.add_argument('--update_set', type=int, default=0)  # LeafInfluence

    parser.add_argument('--similarity', type=str, default='dot_prod')  # Similarity

    parser.add_argument('--kernel', type=str, default='lpw')  # Trex & similarity
    parser.add_argument('--target', type=str, default='actual')  # Trex
    parser.add_argument('--lmbd', type=float, default=0.003)  # Trex
    parser.add_argument('--n_epoch', type=str, default=3000)  # Trex

    parser.add_argument('--trunc_frac', type=float, default=0.25)  # DShap
    parser.add_argument('--check_every', type=int, default=100)  # DShap

    parser.add_argument('--sub_frac', type=float, default=0.7)  # SubSample
    parser.add_argument('--n_iter', type=int, default=4000)  # SubSample

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    # Additional settings
    parser.add_argument('--n_repeat', type=int, default=5)

    args = parser.parse_args()
    main(args)
