"""
Compute global influence.
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
sys.path.insert(0, here + '/../../')
import intent
import util


def remove_and_evaluate(inf_obj, objective, ranking, tree,
                        X_train, y_train, X_test, y_test,
                        remove_frac, n_ckpt, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_pred

    # get list of remove fractions
    remove_frac_arr = np.linspace(0, remove_frac, n_ckpt + 1)[1:]

    # result container
    result = {}
    result['remove_frac'] = remove_frac_arr
    result['loss'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)
    result['acc'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)
    result['auc'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)

    res = eval_fn(objective, tree, X_test, y_test, logger, prefix=f'Ckpt. {0:>5}: {0:>5.2f}%')
    result['loss'][0] = res['loss']
    result['acc'][0] = res['acc']
    result['auc'][0] = res['auc']

    for i in range(n_ckpt):

        remove_frac = remove_frac_arr[i]
        n_remove = int(X_train.shape[0] * remove_frac)

        new_X_train = np.delete(X_train, ranking[:n_remove], axis=0)
        new_y_train = np.delete(y_train, ranking[:n_remove])

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            new_tree = clone(tree).fit(new_X_train, new_y_train)

            prefix = f'Ckpt. {i + 1:>5}: {remove_frac * 100:>5.2f}%'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)
            result['loss'][i] = res['loss']
            result['acc'][i] = res['acc']
            result['auc'][i] = res['auc']

    return result


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

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s\n')

    # compute influence
    start2 = time.time()

    if 'test' in args.strategy:
        local_influence = explainer.get_local_influence(X_test, y_test)  # shape=(no. test,)

        # aggregate local influences
        if args.strategy == 'test_sum':
            influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

        elif args.strategy == 'test_mean':
            influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

        elif args.strategy == 'test_abs_sum':
            influence = np.sum(np.abs(local_influence), axis=1)  # shape=(no. train,)

        else:
            assert args.strategy == 'test_abs_mean'
            influence = np.sum(np.abs(local_influence), axis=1)  # shape=(no. train,)

    else:
        assert 'self' in args.strategy

        # select batch size
        batch_size = 100

        if args.method in ['random', 'minority', 'loo', 'subsample']:
            batch_size = X_train.shape[0]

        influence = explainer.get_self_influence(X_train, y_train, batch_size=batch_size)  # shape=(no. train,)

        if args.strategy == 'self_abs':
            influence = np.abs(influence)

    inf_time = time.time() - start2
    logger.info(f'[INFO] explainer influence time: {inf_time:.5f}s\n')

    # get ranking
    ranking = np.argsort(influence)[::-1]  # shape=(no. train,)

    res = remove_and_evaluate(args.inf_obj, objective, ranking, tree,
                              X_train, y_train, X_test, y_test,
                              args.remove_frac, args.n_ckpt, logger)

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['influence'] = influence
    result['remove_frac'] = res['remove_frac']  # shape=(no. ckpts,)
    result['loss'] = res['loss']  # shape=(no. ckpts,)
    result['acc'] = res['acc']  # shape=(no. ckpts,)
    result['auc'] = res['auc']  # shape=(no. ckpts,)
    result['ranking'] = ranking
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
    exp_dict = {'inf_obj': args.inf_obj, 'strategy': args.strategy,
                'remove_frac': args.remove_frac, 'n_ckpt': args.n_ckpt}
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
                           f'{args.method}_{hash_str}')

    # create output directory and clear previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(f'\ntimestamp: {datetime.now()}')

    experiment(args, logger, params, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out_dir', type=str, default='output/influence2/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='global')
    parser.add_argument('--strategy', type=str, default='test_sum')
    parser.add_argument('--remove_frac', type=float, default=0.5)
    parser.add_argument('--n_ckpt', type=int, default=50)

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

    args = parser.parse_args()
    main(args)
