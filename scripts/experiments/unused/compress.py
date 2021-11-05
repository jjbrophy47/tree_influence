"""
Compute global or local influence.
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
from influence import select_elements


def get_pred(objective, model, X, y):
    """
    Return predictions of shape=(X.shape[0], no. class).
    """

    if objective == 'regression':
        pred = model.predict(X)

    elif objective == 'binary':
        pred = model.predict_proba(X)[:, 1]

    elif objective == 'multiclass':
        pred = model.predict_proba(X)

    else:
        raise ValueError(f'objective {objective} unknown!')

    return pred


def get_ranking(inf_obj, method, params, agg, tree, X_train, y_train, X, y, logger=None):
    """
    Return influence values.
    """

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    if logger:
        logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # compute infu
    start2 = time.time()

    if inf_obj == 'local':
        influence = explainer.get_local_influence(X, y)

        if agg == 'sum':
            influence = np.sum(influence, axis=1)  # shape=(no. train,)

        elif agg == 'abs_sum':
            influence = np.sum(np.abs(influence), axis=1)  # shape=(no. train,)

        elif agg == 'mean':
            influence = np.mean(influence, axis=1)  # shape=(no. train,)

        else:
            assert agg == 'abs_mean'
            influence = np.mean(np.abs(influence), axis=1)  # shape=(no. train,)

    else:
        assert inf_obj == 'global'
        influence = explainer.get_global_influence()

    inf_time = time.time() - start2

    if logger:
        logger.info(f'[INFO] explainer influence time: {inf_time:.5f}s')

    ranking = np.argsort(np.abs(influence))

    return ranking


def remove_and_evaluate(args, objective, params, tree, X_train, y_train, X_test, y_test, test_idxs, logger):

    # initial predictions
    pred = get_pred(objective, tree, X_test, y_test)

    # get list of remove fractions
    remove_frac_arr = np.linspace(0, args.remove_frac, args.n_ckpt + 1)
    n_remove = int(args.remove_frac * X_train.shape[0] / args.n_ckpt)

    # # result container
    # result = {}
    # result['remove_frac'] = remove_frac_arr
    # result['loss'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)
    # result['pred'] = []

    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    new_tree = clone(tree).fit(new_X_train, new_y_train)

    for i in range(1, args.n_ckpt):

        ranking = get_ranking(args.inf_obj, args.method, params, args.agg, new_tree,
                              new_X_train, new_y_train, X_test[test_idxs], y_test[test_idxs], logger=None)

        new_X_train = np.delete(new_X_train, ranking[:n_remove], axis=0)
        new_y_train = np.delete(new_y_train, ranking[:n_remove])

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            new_tree = clone(tree).fit(new_X_train, new_y_train)
            new_pred = get_pred(objective, new_tree, X_test, y_test)

            diff_pred = np.abs(pred - new_pred)
            diff_max = np.max(diff_pred)
            diff_min = np.min(diff_pred)
            diff_avg = np.mean(diff_pred)
            diff_median = np.median(diff_pred)
            diff_std = np.std(diff_pred)
            diff_n_delta = len(np.where(diff_pred > args.delta)[0])

            logger.info(f"[{i:>5}: {remove_frac_arr[i] * 100:>5.2f}%] "
                        f"max.: {diff_max:>5.3f},\tno. > {args.delta}: {diff_n_delta:>10,},\t"
                        f"min.: {diff_min:>5.3f},\t"
                        f"avg.: {diff_avg:>5.3f},\tmedian: {diff_median:>5.3f},\t"
                        f"s.d.: {diff_std:>5.3f}")

    # return result


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

    # ranking = get_ranking(args.inf_obj, args.method, params, agg, tree,
    #                       X_train, y_train, X_test[test_idxs], y_test[test_idxs], logger=logger)

    # # fit explainer
    # start = time.time()
    # explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    # fit_time = time.time() - start - explainer.parse_time_

    # logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s')

    # # compute influence
    # start2 = time.time()

    # if args.inf_obj == 'local':
    #     influence = explainer.get_local_influence(X_test[test_idxs], y_test[test_idxs])

    #     if args.agg == 'sum':
    #         influence = np.sum(influence, axis=1)  # shape=(no. train,)

    #     elif args.agg == 'abs_sum':
    #         influence = np.sum(np.abs(influence), axis=1)  # shape=(no. train,)

    #     elif args.agg == 'mean':
    #         influence = np.mean(influence, axis=1)  # shape=(no. train,)

    #     else:
    #         assert args.agg == 'abs_mean'
    #         influence = np.mean(np.abs(influence), axis=1)  # shape=(no. train,)

    # else:
    #     assert args.inf_obj == 'global'
    #     influence = explainer.get_global_influence()

    # inf_time = time.time() - start2

    # logger.info(f'[INFO] explainer influence time: {inf_time:.5f}s')
    # logger.info(f'[INFO] total time: {time.time() - begin:.5f}s')

    # # get ranking
    # ranking = np.argsort(np.abs(influence))  # least to most influential, shape=(no. train,)

    remove_and_evaluate(args, objective, params, tree, X_train, y_train, X_test, y_test, test_idxs, logger)

    # combine results from each test example
    result['remove_frac'] = res_list[0]['remove_frac']  # shape=(no. ckpts,)
    result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)
    result['pred'] = [res['pred'] for res in res_list]  # shape=(no. test, no. completed ckpts, no class)

    # store ALL train and test predictions
    if objective == 'regression':
        y_train_pred = tree.predict(X_train).reshape(-1, 1)
        y_test_pred = tree.predict(X_test).reshape(-1, 1)

    elif objective == 'binary':
        y_train_pred = tree.predict_proba(X_train)[:, 1].reshape(-1, 1)
        y_test_pred = tree.predict_proba(X_test)[:, 1].reshape(-1, 1)

    else:
        assert objective == 'multiclass'
        y_train_pred = tree.predict_proba(X_train)
        y_test_pred = tree.predict_proba(X_test)

    # save results
    result['influence'] = influence
    result['ranking'] = ranking
    result['test_idxs'] = test_idxs
    result['y_train_pred'] = y_train_pred
    result['y_test_pred'] = y_test_pred
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['fit_time'] = fit_time
    result['inf_time'] = inf_time
    result['total_time'] = time.time() - begin
    result['tree_params'] = tree.get_params()
    result['n_jobs'] = n_jobs
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    # get unique hash for this experiment setting
    exp_dict = {'inf_obj': args.inf_obj, 'n_test': args.n_test,
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
    parser.add_argument('--out_dir', type=str, default='output/compress/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.95)
    parser.add_argument('--n_ckpt', type=int, default=95)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--agg', type=str, default='abs_sum')

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

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    args = parser.parse_args()
    main(args)
