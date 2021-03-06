"""
Model performance.
"""
import os
import sys
import time
import resource
import argparse
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import util


def get_model(args, objective):
    """
    Return the appropriate classifier.
    """
    if args.model in ['lgb', 'sgb', 'cb', 'xgb', 'skgbm', 'skrf']:
        clf = util.get_model(tree_type=args.model,
                             objective=objective,
                             n_tree=args.n_estimators,
                             max_depth=args.max_depth,
                             random_state=args.random_state)
        params = {'n_estimators': [10, 25, 50, 100, 200], 'max_depth': [2, 3, 4, 5, 6, 7]}

        if args.model == 'lgb':
            params['max_depth'] = [-1]
            params['num_leaves'] = [15, 31, 61, 91]

        elif args.model == 'sgb':
            params['max_iter'] = params['n_estimators']
            params['max_depth'] = [None]
            params['max_leaf_nodes'] = [15, 31, 61, 91]
            params['max_bins'] = [50, 100, 250]
            del params['n_estimators']

        elif args.model == 'cb':
            params['learning_rate'] = [0.1, 0.3, 0.6, 0.9]

    elif args.model == 'dt':
        if objective == 'regression':
            clf = DecisionTreeRegressor(random_state=args.random_state)
            params = {'criterion': ['mse', 'friedman_mse'], 'splitter': ['best', 'random'],
                      'max_depth': [3, 5, 10, None]}
        else:
            clf = DecisionTreeClassifier(random_state=args.random_state)
            params = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                      'max_depth': [3, 5, 10, None]}

    elif args.model == 'lr':
        if objective == 'regression':
            clf = Pipeline(steps=[
                ('ss', StandardScaler()),
                ('lr', LinearRegression())
            ])
            params = {}
        else:
            clf = Pipeline(steps=[
                ('ss', StandardScaler()),
                ('lr', LogisticRegression(penalty=args.penalty, C=args.C, solver='liblinear',
                                          random_state=args.random_state))
            ])
            params = {'lr__penalty': ['l1', 'l2'], 'lr__C': [1e-2, 1e-1, 1e0]}

    elif args.model == 'svm':
        if objective == 'regression':
            clf = Pipeline(steps=[
                ('ss', StandardScaler()),
                ('svm', SVR(gamma='auto', C=args.C, kernel=args.kernel))
            ])
        else:
            clf = Pipeline(steps=[
                ('ss', StandardScaler()),
                ('svm', SVC(gamma='auto', C=args.C, kernel=args.kernel,
                            probability=True, random_state=args.random_state))
            ])
        params = {'svm__C': [1e-2, 1e-1, 1e0]}

    elif args.model == 'knn':
        if objective == 'regression':
            clf = KNeighborsRegressor(weights=args.weights, n_neighbors=args.n_neighbors)
        else:
            clf = KNeighborsClassifier(weights=args.weights, n_neighbors=args.n_neighbors)
        params = {'n_neighbors': [3, 5, 7, 11, 15, 31, 61]}

    elif args.model == 'mlp':
        if objective == 'regression':
            clf = MLPRegressor()
        else:
            clf = MLPClassifier()
        params = {'hidden_layer_sizes': [(100,), (100, 100)]}

    else:
        raise ValueError('model uknown: {}'.format(args.model))

    return clf, params


def experiment(args, logger, out_dir):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    # start experiment timer
    begin = time.time()

    # pseduo-random number generator
    rng = np.random.default_rng(args.random_state)

    # get data
    X_train, X_test, y_train, y_test, objective = util.get_data(args.data_dir, args.dataset)

    logger.info('no. train: {:,}'.format(X_train.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # tune on a fraction of the training data
    if not args.no_tune:

        if args.tune_frac < 1.0:

            if objective == 'regression':
                n_tune = int(X_train.shape[0] * args.tune_frac)
                tune_indices = rng.choice(X_train.shape[0], size=n_tune, replace=False)

            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                             train_size=args.tune_frac,
                                             random_state=args.random_state)
                tune_indices, _ = list(sss.split(X_train, y_train))[0]

            X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
            logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

        else:
            X_train_sub, y_train_sub = X_train, y_train
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # get model
    model, param_grid = get_model(args, objective)
    logger.info('\nmodel: {}, param_grid: {}'.format(args.model, param_grid))

    # tune the model
    start = time.time()

    if not args.no_tune:

        if objective == 'regression':
            skf = args.cv
            scoring = 'neg_mean_squared_error'

        else:
            scoring = 'accuracy'

            if objective == 'binary':
                pct_neg = len(np.where(y_train == 0)[0]) / len(y_train)
                pct_pos = len(np.where(y_train == 1)[0]) / len(y_train)

                if pct_neg < args.imb_thresh or pct_pos < args.imb_thresh:  # 20% imbalanced or more, use AUC
                    scoring = 'roc_auc'

            skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

        gs = GridSearchCV(model, param_grid, scoring=scoring, cv=skf, verbose=args.verbose)
        gs = gs.fit(X_train_sub, y_train_sub)

        cols = ['mean_fit_time', 'mean_test_score', 'rank_test_score']
        cols += ['param_{}'.format(param) for param in param_grid.keys()]

        df = pd.DataFrame(gs.cv_results_)
        logger.info('\ngridsearch results:')
        logger.info(df[cols].sort_values('rank_test_score'))

        model = clone(gs.best_estimator_)
        logger.info('\nbest params: {}'.format(gs.best_params_))

    tune_time = time.time() - start
    logger.info('\ntune time: {:.3f}s'.format(tune_time))

    # train model
    if args.train_frac < 1.0:
        assert args.train_frac > 0.0
        sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                     train_size=args.train_frac,
                                     random_state=args.random_state)
        train_indices, _ = list(sss.split(X_train, y_train))[0]

        X_train_sub, y_train_sub = X_train[train_indices].copy(), y_train[train_indices].copy()
        logger.info('train instances: {:,}'.format(X_train_sub.shape[0]))

    else:
        X_train_sub, y_train_sub = X_train.copy(), y_train.copy()

    start = time.time()
    model = model.fit(X_train_sub, y_train_sub)
    train_time = time.time() - start
    logger.info('train time: {:.3f}s\n'.format(train_time))

    # evaluate
    res = util.eval_pred(objective, model, X_test, y_test, logger, prefix=args.model)

    if args.dataset == 'naval':  # more precision
        logger.info(f"[{args.model}] "
                    f"acc.: {res['acc']:>10.2e}, "
                    f"AUC: {res['auc']:>10.2e}, "
                    f"loss: {res['loss']:>10.2e}")

    # save results
    result = {}
    result['model'] = args.model
    result['model_params'] = model.get_params()
    result['loss'] = res['loss']
    result['auc'] = res['auc']
    result['acc'] = res['acc']
    result['tune_time'] = tune_time
    result['train_time'] = train_time
    result['max_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB if OSX, GB if Linux
    result['tune_frac'] = args.tune_frac
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # Macs show this in bytes, unix machines show this in KB
    logger.info('\nmax_rss (MB): {:,}'.format(result['max_rss_MB']))
    logger.info('total time: {:.3f}s'.format(time.time() - begin))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           'rs_{}'.format(args.random_state))

    # create outut directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # write everything printed to stdout to this log file
    logfile, stdout, stderr = util.stdout_stderr_to_log(os.path.join(out_dir, 'log+.txt'))

    # run experiment
    experiment(args, logger, out_dir)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output/predictive_performance/')

    # Data settings
    parser.add_argument('--dataset', type=str, default='vaccine')

    # Experiment settings
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--model', type=str, default='lgb')

    # Tuning settings
    parser.add_argument('--no_tune', action='store_true', default=False)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--tune_frac', type=float, default=1.0)
    parser.add_argument('--train_frac', type=float, default=1.0)

    # Tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)

    # LR and SVM hyperparameters
    parser.add_argument('--penalty', type=str, default='l2')
    parser.add_argument('--C', type=float, default=0.1)
    parser.add_argument('--kernel', type=str, default='rbf')

    # KNN hyperparameters
    parser.add_argument('--weights', type=str, default='uniform')
    parser.add_argument('--n_neighbors', type=int, default=3)

    # Extra settings
    parser.add_argument('--imb_thresh', type=float, default=0.21)
    parser.add_argument('--verbose', type=int, default=2)

    args = parser.parse_args()
    main(args)
