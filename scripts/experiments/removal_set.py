"""
Compute influence for a validation set of examples.
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


# def remove_and_evaluate(inf_obj, objective, ranking, tree,
#                         X_train, y_train, X_test, y_test,
#                         remove_frac, n_ckpt, logger):

#     # get appropriate evaluation function
#     eval_fn = util.eval_pred

#     # get list of remove fractions
#     remove_frac_arr = np.linspace(0, remove_frac, n_ckpt + 1)[1:]

#     # result container
#     result = {}
#     result['remove_frac'] = remove_frac_arr
#     result['loss'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)
#     result['acc'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)
#     result['auc'] = np.full(remove_frac_arr.shape[0], np.nan, dtype=np.float32)

#     res = eval_fn(objective, tree, X_test, y_test, logger, prefix=f'Ckpt. {0:>5}: {0:>5.2f}%')
#     result['loss'][0] = res['loss']
#     result['acc'][0] = res['acc']
#     result['auc'][0] = res['auc']

#     for i in range(n_ckpt):

#         remove_frac = remove_frac_arr[i]
#         n_remove = int(X_train.shape[0] * remove_frac)

#         new_X_train = np.delete(X_train, ranking[:n_remove], axis=0)
#         new_y_train = np.delete(y_train, ranking[:n_remove])

#         if objective == 'binary' and len(np.unique(new_y_train)) == 1:
#             logger.info('Only samples from one class remain!')
#             break

#         elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
#             logger.info('At least 1 sample is not present for all classes!')
#             break

#         else:
#             new_tree = clone(tree).fit(new_X_train, new_y_train)

#             prefix = f'Ckpt. {i + 1:>5}: {remove_frac * 100:>5.2f}%'
#             res = eval_fn(objective, new_tree, X_test, y_test, logger, prefix=prefix)
#             result['loss'][i] = res['loss']
#             result['acc'][i] = res['acc']
#             result['auc'][i] = res['auc']

#     return result


def experiment(args, logger, params, out_dir):

    # initialize experiment
    begin = time.time()
    rng = np.random.default_rng(args.random_state)
    result = {}

    # data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    # use a fraction of the test data for validation
    stratify = None if objective == 'regression' else y_test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=1.0 - args.val_frac,
                                                    stratify=stratify, random_state=args.random_state)

    # display dataset statistics
    logger.info(f'\nno. train: {X_train.shape[0]:,}')
    logger.info(f'no. val.: {X_val.shape[0]:,}')
    logger.info(f'no. test: {X_test.shape[0]:,}')
    logger.info(f'no. features: {X_train.shape[1]:,}\n')

    # train tree-ensemble
    hp = util.get_hyperparams(tree_type=args.tree_type, dataset=args.dataset)
    tree = util.get_model(tree_type=args.tree_type, objective=objective, random_state=args.random_state)
    tree.set_params(**hp)

    tree = tree.fit(X_train, y_train)
    res_clean = util.eval_pred(objective, tree, X_test, y_test, logger, prefix='Test (clean)')

    # fit explainer
    start = time.time()
    explainer = intent.TreeExplainer(args.method, params, logger).fit(tree, X_train, y_train)
    fit_time = time.time() - start - explainer.parse_time_

    logger.info(f'\n[INFO] explainer fit time: {fit_time:.5f}s\n')

    # compute influence
    start2 = time.time()

    # aggregate local influences
    local_influence = explainer.get_local_influence(X_val, y_val)  # shape=(no. test,)
    influence = np.sum(local_influence, axis=1)  # shape=(no. train,)

    ranking = np.argsort(influence)[::-1]  # most to least helpful

    inf_time = time.time() - start2
    logger.info(f'[INFO] influence time: {inf_time:.5f}s\n')

    # remove most helpful train examples
    loss = np.full(len(args.remove_frac), np.nan, dtype=np.float32)
    acc = np.full(len(args.remove_frac), np.nan, dtype=np.float32)
    auc = np.full(len(args.remove_frac), np.nan, dtype=np.float32)

    for i, remove_frac in enumerate(args.remove_frac):
        n_remove = int(len(X_train) * remove_frac)
        remove_idxs = ranking[:n_remove]

        # remove examples
        new_X_train = np.delete(X_train, remove_idxs, axis=0)
        new_y_train = np.delete(y_train, remove_idxs)

        # validate new dataset
        if objective == 'binary' and len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        elif objective == 'multiclass' and len(np.unique(new_y_train)) < len(np.unique(y_train)):
            logger.info('At least 1 sample is not present for all classes!')
            break

        # retrain and re-evaluate model on leftover train data
        new_tree = clone(tree).fit(new_X_train, new_y_train)
        res_remove = util.eval_pred(objective, new_tree, X_test, y_test, logger,
                                    prefix=f'Test ({remove_frac * 100:>2.0f}% removal)')

        loss[i] = res_remove['loss']
        acc[i] = res_remove['acc']
        auc[i] = res_remove['auc']

    cum_time = time.time() - begin
    logger.info(f'\n[INFO] total time: {cum_time:.3f}s')

    # save results
    result['influence'] = influence
    result['remove_frac'] = np.array(args.remove_frac, dtype=np.float32)
    result['loss'] = loss  # shape=(no. ckpts,)
    result['acc'] = acc  # shape=(no. ckpts,)
    result['auc'] = auc  # shape=(no. ckpts,)
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
    exp_dict = {'remove_frac': args.remove_frac, 'val_frac': args.val_frac}
    exp_hash = util.dict_to_hash(exp_dict)

    # special cases
    args.leaf_inf_atol = get_special_case_tol(args.dataset, args.tree_type, args.method, args.leaf_inf_atol)

    # get unique hash for the explainer
    params, hash_str = util.explainer_params_to_dict(args.method, vars(args))

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

    # clean up
    util.remove_logger(logger)


if __name__ == '__main__':
    main(exp_args.get_removal_set_args().parse_args())
