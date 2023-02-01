"""
Generates a continuous attribute binary classification dataset.
"""
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.datasets import make_classification

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(random_state=1,
         test_size=0.2,
         n_samples=10000,
         n_features=40,
         n_informative=5,
         n_redundant=5,
         n_repeated=0,
         n_classes=3,
         n_clusters_per_class=2,
         flip_y=0.05):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # retrieve dataset
    data = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_repeated=n_repeated,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               flip_y=flip_y,
                               random_state=random_state)
    X, y = data
    indices = np.arange(len(X))

    rng = np.random.default_rng(random_state)
    train_indices = rng.choice(indices, size=int(len(X) * (1 - test_size)), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    feature = ['x{}'.format(x) for x in range(X_train.shape[1])]

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 'feature': feature}

    _, y_train_cnt = np.unique(y_train, return_counts=True)
    _, y_test_cnt = np.unique(y_test, return_counts=True)
    _, y_cnt = np.unique(y, return_counts=True)

    y_train_dist = y_train_cnt / y_train_cnt.sum()
    y_test_dist = y_test_cnt / y_test_cnt.sum()
    y_dist = y_cnt / y_cnt.sum()

    logger.info(f'train (head): {X_train[:5]}, {y_train[:5]}')
    logger.info(f'test (head): {X_test[:5]}, {y_test[:5]}')
    logger.info(f'feature (head): {feature[:5]}')
    logger.info(f'X_train.shape: {X_train.shape}')
    logger.info(f'X_test.shape: {X_test.shape}')
    logger.info(f'y_train.shape: {y_train.shape}, y_train dist.: {y_train_dist}')
    logger.info(f'y_test.shape: {y_test.shape}, y_test dist.: {y_test_dist}')
    logger.info(f'y.shape: {y.shape}, y dist.: {y_dist}')

    # save to numpy format
    np.save(os.path.join('data.npy'), data)


if __name__ == '__main__':
    main()
