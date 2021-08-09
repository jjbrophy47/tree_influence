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


def remove_and_reinfluence(inf_obj, objective, tree, method, params,
                           X_train, y_train, X_test, y_test,
                           remove_frac, n_ckpt, logger):

    # get appropriate evaluation function
    eval_fn = util.eval_loss

    # get list of remove fractions
    remove_frac_arr = np.linspace(0, remove_frac, n_ckpt + 1)
    n_remove = round((remove_frac * X_train.shape[0]) / n_ckpt)

    # result container
    result = {}
    result['remove_frac'] = remove_frac_arr
    result['loss'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)
    result['ranking'] = np.full(X_train.shape[0], np.nan, dtype=np.int32)
    result['pred'] = []

    # trackers
    indices = np.arange(X_train.shape[0])
    ranking = np.zeros((0,), dtype=np.int32)

    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    new_tree = clone(tree).fit(new_X_train, new_y_train)

    res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=f'{0:>5}: {0:>5.2f}%')
    result['loss'][0] = res['loss']
    result['pred'].append(res['pred'])

    for i in range(1, n_ckpt):

        explainer = intent.TreeExplainer(method, params, logger).fit(new_tree, new_X_train, new_y_train)
        inf = explainer.get_local_influence(X_test, y_test).flatten()

        idxs = np.argsort(inf)[::-1]
        remove_idxs = idxs[:n_remove]

        ranking = np.concatenate([ranking, indices[remove_idxs]])

        new_X_train = np.delete(new_X_train, remove_idxs, axis=0)
        new_y_train = np.delete(new_y_train, remove_idxs)
        indices = np.delete(indices, remove_idxs)

        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        else:
            new_tree = clone(tree).fit(new_X_train, new_y_train)

            remove_frac = remove_frac_arr[i]
            prefix = f'{i:>5}: {remove_frac * 100:>5.2f}%'
            res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)
            result['loss'][i] = res['loss']
            result['pred'].append(res['pred'])

    result['pred'] = np.vstack(result['pred'])
    result['ranking'][:len(ranking)] = ranking

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

    # compute infuence
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    if args.inf_obj == 'global':
        start2 = time.time()
        influence = explainer.get_global_influence(X=X_test, y=y_test)
        inf_time = time.time() - start2

    else:

        # randomly select test instances to compute influence values for
        if args.inf_obj == 'local':
            avail_idxs = np.arange(X_test.shape[0])
            n_test = min(args.n_test, len(avail_idxs))
            test_idxs = select_elements(avail_idxs, rng, n=n_test)

        elif args.inf_obj == 'local_correct' and objective != 'regression':
            y_pred = tree.predict(X_test)
            correct_idxs = np.where(y_pred == y_test)[0]
            n_test = min(args.n_test, len(correct_idxs))
            test_idxs = select_elements(correct_idxs, rng, n=n_test)

        else:
            assert args.inf_obj == 'local_incorrect' and objective != 'regression'
            y_pred = tree.predict(X_test)
            incorrect_idxs = np.where(y_pred != y_test)[0]
            n_test = min(args.n_test, len(incorrect_idxs))
            test_idxs = select_elements(incorrect_idxs, rng, n=n_test)

        # get no. jobs to run in parallel
        if args.n_jobs == -1:
            n_jobs = joblib.cpu_count()

        else:
            assert args.n_jobs >= 1
            n_jobs = min(args.n_jobs, joblib.cpu_count())

        logger.info(f'[INFO] no. jobs: {n_jobs:,}')

        with joblib.Parallel(n_jobs=n_jobs) as parallel:

            n_finish = 0
            n_remain = len(test_idxs)

            res_list = []

            while n_remain > 0:
                n = min(min(10, n_jobs), n_remain)

                results = parallel(joblib.delayed(remove_and_reinfluence)
                                                 (args.inf_obj, objective,
                                                  tree, args.method, params,
                                                  X_train, y_train, X_test[[idx]], y_test[[idx]],
                                                  args.remove_frac, args.n_ckpt, logger)
                                                  for i, idx in enumerate(test_idxs[n_finish: n_finish + n]))

                # synchronization barrier
                res_list += results

                n_finish += n
                n_remain -= n

                cum_time = time.time() - start
                logger.info(f'[INFO] test instances finished: {n_finish:,} / {test_idxs.shape[0]:,}'
                            f', cum. time: {cum_time:.3f}s')

            # combine results from each test example
            result['remove_frac'] = res_list[0]['remove_frac']  # shape=(no. ckpts,)
            result['loss'] = np.vstack([res['loss'] for res in res_list])  # shape=(no. test, no. ckpts)
            result['ranking'] = np.vstack([res['ranking'] for res in res_list])  # shape=(no. test, no. ckpts)
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
    result['test_idxs'] = test_idxs if args.inf_obj != 'global' else ''
    result['y_train_pred'] = y_train_pred
    result['y_test_pred'] = y_test_pred
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
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
    parser.add_argument('--out_dir', type=str, default='output/reinfluence/')

    # Experiment settings
    parser.add_argument('--dataset', type=str, default='surgical')
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--inf_obj', type=str, default='local')
    parser.add_argument('--n_test', type=int, default=100)  # local
    parser.add_argument('--remove_frac', type=float, default=0.05)
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

    parser.add_argument('--n_jobs', type=int, default=-1)  # LOO and DShap
    parser.add_argument('--random_state', type=int, default=1)  # Trex, DShap, random
    parser.add_argument('--global_op', type=str, default='self')  # Trex, loo, DShap

    args = parser.parse_args()
    main(args)
