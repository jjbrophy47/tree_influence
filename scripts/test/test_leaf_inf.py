import os
import sys
import shutil
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import test_util
from tree_influence.explainers import LeafInfluence
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from test_util import _get_model
from test_util import _get_test_data
from test_parser import compare_predictions


def get_cb_influence_original_method(model, X_train, y_train, X_test, y_test, kwargs):
    """
    Compute influence values using the original source.
    """

    update_set = kwargs['update_set']
    k = update_set

    if k == -1:
        update_set = 'AllPoints'

    elif k == 0:
        update_set = 'SinglePoint'

    else:
        update_set = 'TopKLeaves'

    # save CatBoost model
    temp_dir = os.path.join('.catboost_info', 'leaf_influence')
    temp_fp = os.path.join(temp_dir, 'cb.json')
    os.makedirs(temp_dir, exist_ok=True)
    model.save_model(temp_fp, format='json')

    # initialize Leaf Influence
    explainer = CBLeafInfluenceEnsemble(temp_fp,
                                        X_train,
                                        y_train,
                                        k=k,
                                        learning_rate=model.learning_rate_,
                                        update_set=update_set)

    buf = deepcopy(explainer)
    influence = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=np.float32)

    # compute influence for each training instance
    for train_idx in tqdm(range(X_train.shape[0])):
        explainer.fit(removed_point_idx=train_idx, destination_model=buf)
        influence[train_idx, :] = buf.loss_derivative(X_test, y_test)  # shape=(1, no. test)

    # clean up
    shutil.rmtree('.catboost_info')

    return influence


def test_local_influence_binary_original_vs_adapted(args, kwargs, n=10, show_plot=False):
    print(f'\n***** test_local_influence_binary_original_vs_adapted *****')
    args.model_type = 'binary'
    X_train, X_test, y_train, y_test = _get_test_data(args, n_class=2)
    test_ids = np.array([0])

    X_test, y_test = X_train[test_ids], y_train[test_ids]

    tree = _get_model(args)
    tree = tree.fit(X_train, y_train)

    explainer = LeafInfluence(**kwargs).fit(tree, X_train, y_train)

    # compute influences, shape=(no. train, no. test)
    influences1 = explainer.get_local_influence(X_train[test_ids], y_train[test_ids])
    print('finished influence 1...')

    influences2 = get_cb_influence_original_method(tree, X_train, y_train, X_test, y_test, kwargs)
    print('finished influence 2...')

    for i, test_idx in enumerate(test_ids):

        # influence #1
        influence = influences1[:, i]
        s_ids = np.argsort(np.abs(influence))[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:n])
        print('y_train   (head, sorted):', y_train[s_ids][:n])
        print('influence (head, sorted):', influence[s_ids][:n])

        # influence #2
        influence = influences2[:, i]
        s_ids = np.argsort(np.abs(influence))[::-1]

        test_pred = tree.predict_proba(X_train[[test_idx]])[0]
        test_label = y_train[test_idx]

        print(f'\nexplain y_train {test_idx}, pred: {test_pred}, target: {test_label}\n')

        print('sorted indices    (head):', s_ids[:n])
        print('y_train   (head, sorted):', y_train[s_ids][:n])
        print('influence (head, sorted):', influence[s_ids][:n])

    p1 = influences1[:, 0]
    p2 = -influences2[:, 0]
    spearman = spearmanr(p1, p2)[0]
    pearson = pearsonr(p1, p2)[0]
    status = compare_predictions(p1, p2)
    print('spearmanr:', spearman)
    print('pearsonr:', pearson)

    if show_plot:
        plt.scatter(p1, p2)
        plt.show()

    print(f'\n{status}')


def main(args):

    # explainer arguments
    kwargs = {'update_set': args.update_set, 'n_jobs': args.n_jobs}
    kwargs2 = {'update_set': args.update_set, 'atol': args.atol, 'n_jobs': args.n_jobs}

    # tests
    test_util.test_local_influence_regression(args, LeafInfluence, 'leaf_influence', kwargs)
    test_util.test_local_influence_binary(args, LeafInfluence, 'leaf_influence', kwargs)
    test_util.test_local_influence_multiclass(args, LeafInfluence, 'leaf_influence', kwargs2)

    if args.tree_type == 'cb':
        test_local_influence_binary_original_vs_adapted(args, kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data settings
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--n_local', type=int, default=2)
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--n_feat', type=int, default=10)

    # tree-ensemble settings
    parser.add_argument('--n_tree', type=int, default=100)
    parser.add_argument('--n_leaf', type=int, default=31)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--tree_type', type=str, default='lgb')
    parser.add_argument('--model_type', type=str, default='dummy')
    parser.add_argument('--rs', type=int, default=1)

    # explainer settings
    parser.add_argument('--update_set', type=int, default=-1)
    parser.add_argument('--atol', type=float, default=1e-2)
    parser.add_argument('--n_jobs', type=int, default=1)

    args = parser.parse_args()

    main(args)
