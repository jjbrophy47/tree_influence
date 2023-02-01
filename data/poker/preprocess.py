"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main():

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # categorize attributes
    columns = ['C1', 'S1', 'C2', 'S2', 'C3', 'S3', 'C4', 'S4', 'C5', 'S5', 'CLASS']

    # retrieve dataset
    start = time.time()
    train_df = pd.read_csv('poker-hand-training-true.data', header=None, names=columns)
    test_df = pd.read_csv('poker-hand-testing.data', header=None, names=columns)
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['CLASS']
    features['numeric'] = ['C1', 'C2', 'C3', 'C4', 'C5']
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    X_train, y_train, X_test, y_test, feature = util.preprocess(train_df, test_df, features, logger=logger)

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 'feature': feature}

    y = np.concatenate([y_train, y_test])
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
